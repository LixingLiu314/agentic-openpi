"""Re-encode videos with a small keyframe interval for fast random seeking during training.

By default targets aloha_cube (h264, keyframe every ~250 frames) and re-encodes to
libsvtav1 with g=2, matching the banana dataset encoding that pyav can seek efficiently.

Usage:
    uv run python scripts/reencode_videos.py --dataset_dir aloha_cube
    uv run python scripts/reencode_videos.py --dataset_dir aloha_cube --codec libx264 --crf 18
    uv run python scripts/reencode_videos.py --dataset_dir aloha_cube --dry_run

After re-encoding, re-run merge_datasets.py if you use a merged dataset (symlinks will
automatically point to the new files so no re-merge is needed unless you changed file paths).
"""

import argparse
import logging
import os
import tempfile
from pathlib import Path

import av
import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def get_keyframe_interval(video_path: Path, max_check: int = 5) -> float:
    """Return average keyframe interval in seconds (checks first few keyframes)."""
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        pts_list = []
        for packet in container.demux(stream):
            if packet.is_keyframe and packet.pts is not None:
                pts_list.append(float(packet.pts * stream.time_base))
            if len(pts_list) >= max_check:
                break
    if len(pts_list) < 2:
        return float("inf")
    intervals = [pts_list[i + 1] - pts_list[i] for i in range(len(pts_list) - 1)]
    return sum(intervals) / len(intervals)


def reencode_video(src: Path, dst: Path, codec: str, g: int, crf: int) -> None:
    from fractions import Fraction

    codec_options = {"g": str(g), "crf": str(crf)}
    if codec == "libx264":
        codec_options["preset"] = "fast"

    with av.open(str(src)) as inp:
        in_stream = inp.streams.video[0]
        fps = Fraction(in_stream.average_rate)
        width = in_stream.width
        height = in_stream.height

        with av.open(str(dst), "w") as out:
            out_stream = out.add_stream(codec, rate=fps, options=codec_options)
            out_stream.width = width
            out_stream.height = height
            out_stream.pix_fmt = "yuv420p"  # libsvtav1 requires yuv420p

            for frame in inp.decode(in_stream):
                if frame.format.name != "yuv420p":
                    frame = frame.reformat(format="yuv420p")
                for packet in out_stream.encode(frame):
                    out.mux(packet)

            for packet in out_stream.encode():
                out.mux(packet)


def process_dataset(dataset_dir: Path, codec: str, g: int, crf: int,
                    max_interval_threshold: float, dry_run: bool) -> None:
    video_dir = dataset_dir / "videos"
    if not video_dir.exists():
        raise FileNotFoundError(f"No videos/ directory found in {dataset_dir}")

    mp4_files = sorted(video_dir.rglob("*.mp4"))
    # Skip symlinks that point outside this dataset (e.g. merged dataset entries)
    own_files = [f for f in mp4_files if not f.is_symlink()]

    log.info(f"Found {len(own_files)} non-symlink .mp4 files in {dataset_dir}")
    if not own_files:
        log.warning("No non-symlink files found — nothing to re-encode.")
        return

    # Sample a few files to check current keyframe interval
    sample = own_files[::max(1, len(own_files) // 5)][:5]
    avg_intervals = [get_keyframe_interval(f) for f in sample]
    avg_interval = sum(avg_intervals) / len(avg_intervals)
    log.info(f"Current avg keyframe interval (sample): {avg_interval:.3f}s")

    if avg_interval <= max_interval_threshold:
        log.info(f"Keyframe interval {avg_interval:.3f}s ≤ threshold {max_interval_threshold}s — no re-encoding needed.")
        return

    log.info(f"Re-encoding {len(own_files)} files: codec={codec}, g={g}, crf={crf}")
    if dry_run:
        log.info("DRY RUN — no files will be modified.")
        return

    errors = []
    for mp4 in tqdm.tqdm(own_files, desc="re-encoding"):
        tmp_path = mp4.with_suffix(".tmp.mp4")
        try:
            reencode_video(mp4, tmp_path, codec, g, crf)
            tmp_path.replace(mp4)
        except Exception as e:
            log.error(f"Failed to re-encode {mp4}: {e}")
            errors.append(mp4)
            if tmp_path.exists():
                tmp_path.unlink()

    if errors:
        log.warning(f"{len(errors)} files failed to re-encode: {errors}")
    else:
        log.info("All files re-encoded successfully.")

    # Verify one file
    sample_after = own_files[0]
    interval_after = get_keyframe_interval(sample_after)
    log.info(f"Verification — keyframe interval after re-encode: {interval_after:.3f}s (target: {g/30:.3f}s @ 30fps)")


def main():
    parser = argparse.ArgumentParser(description="Re-encode dataset videos for fast seeking.")
    parser.add_argument("--dataset_dir", required=True,
                        help="Dataset root directory (e.g. aloha_cube)")
    parser.add_argument("--codec", default="libsvtav1",
                        choices=["libsvtav1", "libx264"],
                        help="Output codec (default: libsvtav1 to match banana dataset)")
    parser.add_argument("--g", type=int, default=2,
                        help="Keyframe interval in frames (default: 2, matches banana)")
    parser.add_argument("--crf", type=int, default=30,
                        help="CRF quality (default: 30 for libsvtav1; use 18-23 for libx264)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Skip re-encoding if avg keyframe interval is already <= this (seconds)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print what would be done without modifying any files")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = Path.cwd() / dataset_dir

    process_dataset(dataset_dir, args.codec, args.g, args.crf, args.threshold, args.dry_run)


if __name__ == "__main__":
    main()
