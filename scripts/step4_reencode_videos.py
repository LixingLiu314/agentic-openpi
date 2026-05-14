"""
Step 4 — Re-encode videos H264→AV1 and make binary dataset fully independent (no symlinks):
  1. Re-encode videos: source H264 → binary dataset AV1 (libsvtav1, g=2, crf=30)
  2. Copy data_quality/ and trajectory_data/ from original
  3. Update meta/info.json codec fields

Run from the agentic-openpi root:
  python Datasets/make_binary_dataset_independent.py
"""

import json
import logging
import pathlib
import shutil
from fractions import Fraction

import av
import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

SRC = pathlib.Path("Datasets/avoid_obstable/aloha_banana_obstacle_gripper_binary")
DST = pathlib.Path("Datasets/avoid_obstable/aloha_banana_obstacle_gripper_binary")

CODEC = "libsvtav1"
G     = 2    # keyframe every 2 frames → fast random seeking
CRF   = 30


def reencode(src: pathlib.Path, dst: pathlib.Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.mp4")
    try:
        with av.open(str(src)) as inp:
            in_stream = inp.streams.video[0]
            fps    = Fraction(in_stream.average_rate)
            width  = in_stream.width
            height = in_stream.height

            with av.open(str(tmp), "w") as out:
                out_stream = out.add_stream(CODEC, rate=fps,
                                            options={"g": str(G), "crf": str(CRF)})
                out_stream.width   = width
                out_stream.height  = height
                out_stream.pix_fmt = "yuv420p"

                for frame in inp.decode(in_stream):
                    if frame.format.name != "yuv420p":
                        frame = frame.reformat(format="yuv420p")
                    for pkt in out_stream.encode(frame):
                        out.mux(pkt)
                for pkt in out_stream.encode():
                    out.mux(pkt)

        tmp.replace(dst)
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"Re-encode failed {src}: {e}") from e


def break_symlink_copy(name: str) -> None:
    link = DST / name
    if link.is_symlink():
        link.unlink()
        log.info(f"Removed symlink {name}/")
    dst_dir = DST / name
    src_dir = SRC / name
    if not dst_dir.exists() and src_dir.exists():
        shutil.copytree(src_dir, dst_dir)
        log.info(f"Copied {name}/  ({sum(1 for _ in dst_dir.rglob('*'))} items)")


def main() -> None:
    # ── 1. Re-encode videos (H264 → AV1) ─────────────────────────────────────
    vid_link = DST / "videos"
    if vid_link.is_symlink():
        vid_link.unlink()
        log.info("Removed videos/ symlink")

    src_vids = sorted(SRC.glob("videos/chunk-*/*/episode_*.mp4"))
    log.info(f"Re-encoding {len(src_vids)} videos  ({CODEC}, g={G}, crf={CRF}) ...")

    errors = []
    for src_mp4 in tqdm.tqdm(src_vids, desc="encode"):
        rel     = src_mp4.relative_to(SRC / "videos")
        dst_mp4 = DST / "videos" / rel
        try:
            reencode(src_mp4, dst_mp4)
        except RuntimeError as e:
            log.error(str(e))
            errors.append(src_mp4)

    if errors:
        log.error(f"{len(errors)} videos failed: {errors}")
    else:
        log.info("All videos re-encoded.")

    # ── 2. Copy data_quality/ and trajectory_data/ ────────────────────────────
    for name in ("data_quality", "trajectory_data"):
        break_symlink_copy(name)

    # ── 3. Update meta/info.json codec fields ─────────────────────────────────
    info_path = DST / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    for key, feat in info.get("features", {}).items():
        if feat.get("dtype") == "video":
            feat.setdefault("video_info", {})["video.codec"] = "av1"
            feat.setdefault("info",       {})["video.codec"] = "av1"
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2))
    log.info("Updated meta/info.json codec → av1")

    # ── 4. Quick verification ─────────────────────────────────────────────────
    dst_vids = sorted(DST.glob("videos/chunk-*/*/episode_*.mp4"))
    log.info(f"Videos in binary dataset: {len(dst_vids)}")
    if dst_vids:
        with av.open(str(dst_vids[0])) as c:
            codec_name = c.streams.video[0].codec_context.name
        log.info(f"Sample codec: {codec_name}  (file: {dst_vids[0].name})")

    # Check no symlinks remain
    remaining = [p for p in DST.iterdir() if p.is_symlink()]
    log.info(f"Remaining symlinks: {remaining if remaining else 'none'}")
    log.info("Done.")


if __name__ == "__main__":
    main()
