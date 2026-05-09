#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm


def parse_episode_tokens(values: list[str] | None) -> set[int]:
    selected: set[int] = set()
    if not values:
        return selected
    for raw_value in values:
        for token in raw_value.replace(",", " ").split():
            if ":" in token:
                parts = token.split(":")
                if len(parts) != 2:
                    raise ValueError(f"Invalid episode range {token!r}; expected start:stop")
                if parts[0] == "" or parts[1] == "":
                    raise ValueError(f"Open-ended ranges are not supported here: {token!r}")
                selected.update(range(int(parts[0]), int(parts[1])))
            else:
                selected.add(int(token))
    return selected


def load_available_episodes(traj_json: Path) -> list[int]:
    with traj_json.expanduser().open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return sorted(int(result["episode_index"]) for result in payload.get("results", []))


def render_one(command: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return proc.returncode, proc.stdout, proc.stderr


def add_bool_argument(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    parser.add_argument(name, dest=name.lstrip("-").replace("-", "_"), action="store_true", help=help_text)
    parser.add_argument(
        f"--no-{name.lstrip('-')}",
        dest=name.lstrip("-").replace("-", "_"),
        action="store_false",
        help=f"Disable: {help_text}",
    )
    parser.set_defaults(**{name.lstrip("-").replace("-", "_"): default})


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch-render annotated Aloha FK trajectory videos.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("playground/Datasets/aloha_banana_lerobot_absolute"),
    )
    parser.add_argument("--traj-json", type=Path, required=True)
    parser.add_argument("--episodes", nargs="*", default=None, help="Episode ids or ranges, e.g. 0 10:20. Default: all in traj-json.")
    parser.add_argument("--exclude-episodes", nargs="*", default=None, help="Episode ids or ranges to skip.")
    parser.add_argument("--arm", choices=["left", "right", "both"], default="both")
    parser.add_argument("--video-key", default="observation.images.cam_high")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--codec", default="libx264")
    parser.add_argument("--future-frames", type=int, default=60)
    parser.add_argument("--point-stride", type=int, default=10)
    parser.add_argument("--line-width", type=int, default=3)
    parser.add_argument("--marker-radius", type=int, default=7)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    add_bool_argument(parser, "--skip-existing", default=True, help_text="Skip videos that already exist.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    dataset_path = args.dataset_path.expanduser()
    traj_json = args.traj_json.expanduser()
    available = load_available_episodes(traj_json)

    selected = parse_episode_tokens(args.episodes)
    episodes = sorted(selected.intersection(available)) if selected else available
    excluded = parse_episode_tokens(args.exclude_episodes)
    episodes = [episode for episode in episodes if episode not in excluded]

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = dataset_path / "trajectory_data" / "visualizations_all"
    output_dir = output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    script_path = Path(__file__).with_name("visualize_aloha_fk_trajectory.py")
    commands = []
    skipped = 0
    for episode in episodes:
        output_path = output_dir / f"episode_{episode:06d}_{args.arm}_trajectory.mp4"
        if args.skip_existing and output_path.exists():
            skipped += 1
            continue
        command = [
            sys.executable,
            str(script_path),
            "--dataset-path",
            str(dataset_path),
            "--traj-json",
            str(traj_json),
            "--episode",
            str(episode),
            "--arm",
            args.arm,
            "--video-key",
            args.video_key,
            "--output",
            str(output_path),
            "--codec",
            args.codec,
            "--future-frames",
            str(args.future_frames),
            "--point-stride",
            str(args.point_stride),
            "--line-width",
            str(args.line_width),
            "--marker-radius",
            str(args.marker_radius),
        ]
        if args.max_frames is not None:
            command.extend(["--max-frames", str(args.max_frames)])
        commands.append(command)

    print(f"Rendering {len(commands)} videos to {output_dir} (skipped existing: {skipped})")
    failures = []
    if args.num_workers <= 1:
        for command in tqdm(commands, desc="Rendering videos"):
            code, stdout, stderr = render_one(command)
            if code != 0:
                failures.append({"command": command, "returncode": code, "stdout": stdout, "stderr": stderr})
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(render_one, command): command for command in commands}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Rendering videos"):
                command = futures[future]
                code, stdout, stderr = future.result()
                if code != 0:
                    failures.append({"command": command, "returncode": code, "stdout": stdout, "stderr": stderr})

    report = {
        "traj_json": str(traj_json),
        "output_dir": str(output_dir),
        "arm": args.arm,
        "num_requested": len(episodes),
        "num_rendered": len(commands) - len(failures),
        "num_skipped_existing": skipped,
        "num_failed": len(failures),
        "failures": failures,
    }
    report_path = output_dir / "render_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved render report to {report_path}")
    if failures:
        raise SystemExit(f"{len(failures)} videos failed; see {report_path}")


if __name__ == "__main__":
    main()
