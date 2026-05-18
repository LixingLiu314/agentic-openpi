"""Dataset-image retrieval for trajectory reference suggestions.

The GUI runtime path is intentionally small:

1. Load a precomputed torch tensor cache once.
2. Encode the current ``cam_high`` frame with the same lightweight embedder.
3. Run one cosine-similarity matrix-vector product and return the Top-1
   training trajectory text.

The expensive part, decoding every dataset video frame, happens offline via
``build_retrieval_cache`` or ``tools/trajectory/build_trajectory_retrieval_cache.py``.
"""
from __future__ import annotations

from collections.abc import Iterable
import dataclasses
import json
import logging
import pathlib
import re
import time
from typing import Any

import cv2
import numpy as np
import torch

from .doubao_predictor import coords_to_loc_tokens

logger = logging.getLogger(__name__)


SCHEMA_VERSION = 1
DEFAULT_CAMERA_KEY = "observation.images.cam_high"
DEFAULT_TRAJ_COLUMN = "traj_cot"
DEFAULT_CACHE_FILENAME = "cam_high_traj_reference_cache.pt"

_EPISODE_RE = re.compile(r"episode_(\d+)\.parquet$")
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)


@dataclasses.dataclass(frozen=True)
class RetrievalResult:
    """Top-1 trajectory retrieval result for one live frame."""

    trajectory_text: str
    score: float
    episode_index: int
    frame_index: int
    search_ms: float
    cache_path: str


class LightweightImageEmbedder:
    """Fast deterministic image embedder used for cache and runtime search.

    This deliberately avoids downloading CLIP/ResNet weights inside the robot
    GUI process. The feature vector combines low-frequency DCT structure,
    Sobel edge layout, and RGB color histograms, then L2-normalizes the result.
    It is not a semantic foundation-model embedding, but it is stable,
    dependency-light, and fast enough for sub-second Top-1 search over a local
    tensor cache. A CLIP/ResNet encoder can be swapped in later as long as it
    produces the same normalized ``N x D`` cache contract.
    """

    name = "lightweight-dct-edge-hist-v1"

    def __init__(
        self,
        *,
        image_size: int = 64,
        dct_size: int = 16,
        edge_grid: int = 8,
        hist_bins: int = 16,
    ) -> None:
        self.image_size = int(image_size)
        self.dct_size = int(dct_size)
        self.edge_grid = int(edge_grid)
        self.hist_bins = int(hist_bins)
        if min(self.image_size, self.dct_size, self.edge_grid, self.hist_bins) <= 0:
            raise ValueError("embedder sizes must be positive")
        if self.dct_size > self.image_size:
            raise ValueError("dct_size cannot exceed image_size")

    @property
    def config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "image_size": self.image_size,
            "dct_size": self.dct_size,
            "edge_grid": self.edge_grid,
            "hist_bins": self.hist_bins,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> LightweightImageEmbedder:
        config = dict(config or {})
        name = config.pop("name", cls.name)
        if name != cls.name:
            raise ValueError(f"Unsupported trajectory retrieval embedder: {name!r}")
        allowed = {"image_size", "dct_size", "edge_grid", "hist_bins"}
        return cls(**{k: v for k, v in config.items() if k in allowed})

    def encode_one(self, image_hwc_rgb: np.ndarray) -> torch.Tensor:
        arr = _as_hwc_uint8_rgb(image_hwc_rgb)
        resized = cv2.resize(
            arr,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        )
        rgb = resized.astype(np.float32) / 255.0
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        dct = cv2.dct(gray)
        dct_low = dct[: self.dct_size, : self.dct_size].reshape(-1)

        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edges = cv2.magnitude(sobel_x, sobel_y)
        edge_grid = cv2.resize(
            edges,
            (self.edge_grid, self.edge_grid),
            interpolation=cv2.INTER_AREA,
        ).reshape(-1)

        hist_parts = []
        for channel in range(3):
            hist, _ = np.histogram(
                rgb[:, :, channel],
                bins=self.hist_bins,
                range=(0.0, 1.0),
                density=False,
            )
            hist = hist.astype(np.float32)
            denom = float(hist.sum())
            if denom > 0:
                hist /= denom
            hist_parts.append(hist)

        feat = np.concatenate([dct_low, edge_grid, *hist_parts]).astype(np.float32)
        feat -= float(feat.mean())
        norm = float(np.linalg.norm(feat))
        if norm > 1e-12:
            feat /= norm
        return torch.from_numpy(feat)

    def encode_batch(self, images_hwc_rgb: Iterable[np.ndarray]) -> torch.Tensor:
        encoded = [self.encode_one(image) for image in images_hwc_rgb]
        if not encoded:
            return torch.empty((0, self.feature_dim), dtype=torch.float32)
        return torch.stack(encoded, dim=0).contiguous()

    @property
    def feature_dim(self) -> int:
        return self.dct_size * self.dct_size + self.edge_grid * self.edge_grid + 3 * self.hist_bins


class TrajectoryReferenceRetriever:
    """Loads a precomputed cache and performs sub-second Top-1 search."""

    def __init__(self, cache_path: str | pathlib.Path, *, device: str = "cpu") -> None:
        self.cache_path = pathlib.Path(cache_path).expanduser()
        if not self.cache_path.exists():
            raise FileNotFoundError(self.cache_path)
        payload = _torch_load_cache(self.cache_path)
        schema_version = int(payload.get("schema_version", 0))
        if schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported trajectory retrieval cache schema {schema_version}; "
                f"expected {SCHEMA_VERSION}"
            )
        self.embedder = LightweightImageEmbedder.from_config(payload.get("embedder"))
        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
            raise ValueError(f"Invalid embeddings tensor in {self.cache_path}")
        if embeddings.shape[1] != self.embedder.feature_dim:
            raise ValueError(
                f"Cache feature dimension {embeddings.shape[1]} does not match "
                f"embedder dimension {self.embedder.feature_dim}"
            )
        refs = payload.get("refs")
        if not isinstance(refs, list) or len(refs) != int(embeddings.shape[0]):
            raise ValueError("Cache refs must be a list with one item per embedding")

        self.embeddings = embeddings.to(device=device, dtype=torch.float32).contiguous()
        self.refs = refs
        self.metadata = dict(payload.get("metadata") or {})
        self.device = torch.device(device)

    @property
    def count(self) -> int:
        return int(self.embeddings.shape[0])

    def suggest(self, image_hwc_rgb: np.ndarray) -> RetrievalResult | None:
        if self.count == 0:
            return None
        started = time.perf_counter()
        query = self.embedder.encode_one(image_hwc_rgb).to(device=self.device, dtype=torch.float32)
        with torch.no_grad():
            scores = torch.mv(self.embeddings, query)
            score, index = torch.max(scores, dim=0)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        ref = self.refs[int(index.item())]
        text = str(ref.get("trajectory_text", "")).strip()
        if not text:
            return None
        if elapsed_ms > 1000.0:
            logger.warning("Trajectory reference retrieval took %.1f ms (>1s target)", elapsed_ms)
        return RetrievalResult(
            trajectory_text=text,
            score=float(score.item()),
            episode_index=int(ref.get("episode_index", -1)),
            frame_index=int(ref.get("frame_index", -1)),
            search_ms=elapsed_ms,
            cache_path=str(self.cache_path),
        )


def default_cache_path(dataset_root: str | pathlib.Path) -> pathlib.Path:
    return pathlib.Path(dataset_root).expanduser() / "trajectory_data" / DEFAULT_CACHE_FILENAME


def build_retrieval_cache(
    dataset_root: str | pathlib.Path,
    *,
    output_path: str | pathlib.Path | None = None,
    camera_key: str = DEFAULT_CAMERA_KEY,
    traj_column: str = DEFAULT_TRAJ_COLUMN,
    cot_json_path: str | pathlib.Path | None = None,
    frame_stride: int = 1,
    batch_size: int = 256,
    max_frames: int | None = None,
    progress: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build and save an embedding cache for all dataset main-camera frames."""
    dataset_root = pathlib.Path(dataset_root).expanduser().resolve()
    if output_path is None:
        output_path = default_cache_path(dataset_root)
    output_path = pathlib.Path(output_path).expanduser()
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Cache already exists: {output_path}. Pass overwrite=True to replace it.")

    frame_stride = max(1, int(frame_stride))
    batch_size = max(1, int(batch_size))
    info = _load_info(dataset_root)
    chunks_size = int(info.get("chunks_size", 1000))
    cot_prompts = _load_cot_prompts(dataset_root, cot_json_path)
    embedder = LightweightImageEmbedder()

    parquet_paths = sorted((dataset_root / "data").glob("chunk-*/episode_*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No episode parquet files found under {dataset_root / 'data'}")

    iterator = _maybe_tqdm(parquet_paths, enabled=progress, desc="episodes")
    tensors: list[torch.Tensor] = []
    refs: list[dict[str, Any]] = []
    batch_images: list[np.ndarray] = []
    batch_refs: list[dict[str, Any]] = []
    skipped = {"missing_video": 0, "missing_traj": 0, "decode_errors": 0}

    def flush_batch() -> None:
        nonlocal batch_images, batch_refs
        if not batch_images:
            return
        tensors.append(embedder.encode_batch(batch_images))
        refs.extend(batch_refs)
        batch_images = []
        batch_refs = []

    for parquet_path in iterator:
        if max_frames is not None and len(refs) >= int(max_frames):
            break
        episode_index = _episode_index_from_parquet(parquet_path)
        frame_rows = _load_episode_rows(
            parquet_path,
            episode_index=episode_index,
            camera_key=camera_key,
            traj_column=traj_column,
            cot_prompts=cot_prompts,
        )
        if not frame_rows:
            continue
        first_row = next(iter(frame_rows.values()))
        video_path = _resolve_video_path(
            dataset_root,
            episode_index=episode_index,
            chunks_size=chunks_size,
            camera_key=camera_key,
            row_video_path=first_row.get("video_path"),
        )
        if not video_path.exists():
            skipped["missing_video"] += len(frame_rows)
            logger.warning("Skipping episode %06d; video not found: %s", episode_index, video_path)
            continue
        try:
            for frame_index, frame in _iter_video_frames(video_path):
                if max_frames is not None and len(refs) + len(batch_refs) >= int(max_frames):
                    break
                if frame_index % frame_stride != 0:
                    continue
                row = frame_rows.get(frame_index)
                if row is None:
                    continue
                text = str(row.get("trajectory_text", "")).strip()
                if not text:
                    skipped["missing_traj"] += 1
                    continue
                batch_images.append(frame)
                batch_refs.append(
                    {
                        "episode_index": int(episode_index),
                        "frame_index": int(frame_index),
                        "trajectory_text": text,
                        "video_path": str(video_path),
                    }
                )
                if len(batch_images) >= batch_size:
                    flush_batch()
        except Exception as e:
            skipped["decode_errors"] += 1
            logger.warning("Skipping episode %06d after video decode error: %s", episode_index, e)
        finally:
            flush_batch()

    if not tensors:
        raise RuntimeError("No trajectory retrieval embeddings were produced.")

    embeddings = torch.cat(tensors, dim=0).contiguous()
    embeddings = torch.nn.functional.normalize(embeddings.to(dtype=torch.float32), p=2, dim=1)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_root": str(dataset_root),
            "camera_key": camera_key,
            "traj_column": traj_column,
            "cot_json_path": "" if cot_json_path is None else str(pathlib.Path(cot_json_path).expanduser()),
            "frame_stride": int(frame_stride),
            "num_embeddings": int(embeddings.shape[0]),
            "num_refs": len(refs),
            "skipped": skipped,
        },
        "embedder": embedder.config,
        "embeddings": embeddings,
        "refs": refs,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return {
        "cache_path": str(output_path),
        "num_embeddings": int(embeddings.shape[0]),
        "feature_dim": int(embeddings.shape[1]),
        "skipped": skipped,
    }


def _torch_load_cache(path: pathlib.Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _as_hwc_uint8_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _clean_trajectory_text(text: Any) -> str:
    text = str(text or "").strip()
    if not text:
        return ""
    text = _BR_RE.sub(" ", text)
    text = coords_to_loc_tokens(text)
    return " ".join(text.split())


def _load_info(dataset_root: pathlib.Path) -> dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        return {}
    return json.loads(info_path.read_text(encoding="utf-8"))


def _load_cot_prompts(
    dataset_root: pathlib.Path,
    cot_json_path: str | pathlib.Path | None,
) -> dict[str, dict[str, str]]:
    path = pathlib.Path(cot_json_path).expanduser() if cot_json_path is not None else (
        dataset_root / "trajectory_data" / "cot_text_prompts.json"
    )
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    prompts = payload.get("prompts", {})
    if not isinstance(prompts, dict):
        return {}
    return {
        str(ep): {str(frame): _clean_trajectory_text(text) for frame, text in frames.items()}
        for ep, frames in prompts.items()
        if isinstance(frames, dict)
    }


def _episode_index_from_parquet(path: pathlib.Path) -> int:
    match = _EPISODE_RE.search(path.name)
    if not match:
        raise ValueError(f"Could not parse episode index from {path}")
    return int(match.group(1))


def _load_episode_rows(
    parquet_path: pathlib.Path,
    *,
    episode_index: int,
    camera_key: str,
    traj_column: str,
    cot_prompts: dict[str, dict[str, str]],
) -> dict[int, dict[str, Any]]:
    import pyarrow.parquet as pq

    schema_names = set(pq.read_schema(parquet_path).names)
    columns = [name for name in ("episode_index", "frame_index", camera_key, traj_column) if name in schema_names]
    if "frame_index" not in columns:
        raise KeyError(f"{parquet_path} does not contain a frame_index column")
    table = pq.read_table(parquet_path, columns=columns)
    rows = table.to_pylist()
    by_frame: dict[int, dict[str, Any]] = {}
    fallback_prompts = cot_prompts.get(str(episode_index), {})
    for i, row in enumerate(rows):
        frame_index = int(row.get("frame_index", i))
        text = _clean_trajectory_text(row.get(traj_column, "")) if traj_column in row else ""
        if not text:
            text = fallback_prompts.get(str(frame_index), "")
        by_frame[frame_index] = {
            "trajectory_text": text,
            "video_path": row.get(camera_key, ""),
        }
    return by_frame


def _resolve_video_path(
    dataset_root: pathlib.Path,
    *,
    episode_index: int,
    chunks_size: int,
    camera_key: str,
    row_video_path: Any,
) -> pathlib.Path:
    if row_video_path:
        candidate = pathlib.Path(str(row_video_path)).expanduser()
        if candidate.is_absolute():
            return candidate
        candidate = dataset_root / candidate
        if candidate.exists():
            return candidate
    chunk_index = episode_index // max(1, chunks_size)
    return dataset_root / "videos" / f"chunk-{chunk_index:03d}" / camera_key / f"episode_{episode_index:06d}.mp4"


def _iter_video_frames(video_path: pathlib.Path):
    import av

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame_index, frame in enumerate(container.decode(stream)):
            yield frame_index, frame.to_ndarray(format="rgb24")


def _maybe_tqdm(items, *, enabled: bool, desc: str):
    if not enabled:
        return items
    try:
        from tqdm import tqdm
    except Exception:
        return items
    return tqdm(items, desc=desc)


__all__ = [
    "DEFAULT_CACHE_FILENAME",
    "DEFAULT_CAMERA_KEY",
    "DEFAULT_TRAJ_COLUMN",
    "LightweightImageEmbedder",
    "RetrievalResult",
    "TrajectoryReferenceRetriever",
    "build_retrieval_cache",
    "default_cache_path",
]
