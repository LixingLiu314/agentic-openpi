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
            if ":" not in token:
                selected.add(int(token))
                continue
            parts = token.split(":")
            if len(parts) != 2:
                raise ValueError(f"Invalid episode range {token!r}; expected start:stop")
            if parts[0] == "" or parts[1] == "":
                raise ValueError(f"Open-ended ranges are not supported here: {token!r}")
            selected.update(range(int(parts[0]), int(parts[1])))
    return selected


def load_available_episodes(dataset_path: Path) -> list[int]:
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        info_path = dataset_path / "meta" / "info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        return list(range(int(info["num_episodes"])))
    episodes: list[int] = []
    with episodes_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            episodes.append(int(item["episode_index"]))
    return sorted(episodes)


def render_one(command: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return proc.returncode, proc.stdout, proc.stderr


def add_bool_argument(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    dest = name.lstrip("-").replace("-", "_")
    parser.add_argument(name, dest=dest, action="store_true", help=help_text)
    parser.add_argument(f"--no-{name.lstrip('-')}", dest=dest, action="store_false", help=f"Disable: {help_text}")
    parser.set_defaults(**{dest: default})


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch-render prompt-point trajectory overlay videos.")
    parser.add_argument("--dataset-path", type=Path, default=Path("playground/Datasets/aloha_letter"))
    parser.add_argument("--episodes", nargs="*", default=None, help="Episode ids or ranges, e.g. 0 10:20. Default: all.")
    parser.add_argument("--exclude-episodes", nargs="*", default=None, help="Episode ids or ranges to skip.")
    parser.add_argument("--cot-json", type=Path, default=None)
    parser.add_argument("--prompt-source", choices=["auto", "cot-json", "parquet"], default="auto")
    parser.add_argument("--traj-column", default="traj_cot")
    parser.add_argument("--frame-policy", choices=["direct-window", "eval-blocking", "per-frame"], default="direct-window")
    parser.add_argument("--update-every", type=int, default=60)
    parser.add_argument("--video-key", default="observation.images.cam_high")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--codec", default="libx264")
    parser.add_argument("--max-video-frames", type=int, default=None)
    parser.add_argument("--coord-range", type=int, default=1000)
    parser.add_argument("--loc-range", type=int, default=1000)
    parser.add_argument("--marker-radius", type=int, default=8)
    parser.add_argument("--line-width", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--show-prompt-text", action="store_true")
    add_bool_argument(parser, "--skip-existing", default=True, help_text="Skip videos that already exist.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    dataset_path = args.dataset_path.expanduser()
    episodes_available = load_available_episodes(dataset_path)

    selected = parse_episode_tokens(args.episodes)
    episodes = sorted(selected.intersection(episodes_available)) if selected else episodes_available
    excluded = parse_episode_tokens(args.exclude_episodes)
    episodes = [episode for episode in episodes if episode not in excluded]

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = dataset_path / "trajectory_data" / "prompt_point_visualizations_all" / args.frame_policy
    output_dir = output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = output_dir / "_episode_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    cot_json = args.cot_json
    if cot_json is None:
        cot_json = dataset_path / "trajectory_data" / "cot_text_prompts.json"
    cot_json = cot_json.expanduser()

    script_path = Path(__file__).with_name("visualize_traj_prompt_points.py")
    commands: list[list[str]] = []
    skipped = 0
    for episode in episodes:
        output_path = output_dir / f"episode_{episode:06d}_{args.frame_policy}_prompt_points.mp4"
        if args.skip_existing and output_path.exists() and output_path.stat().st_size > 0:
            skipped += 1
            continue
        episode_report_dir = reports_dir / f"episode_{episode:06d}"
        command = [
            sys.executable,
            str(script_path),
            "--dataset-path",
            str(dataset_path),
            "--episode",
            str(episode),
            "--cot-json",
            str(cot_json),
            "--prompt-source",
            args.prompt_source,
            "--traj-column",
            args.traj_column,
            "--frame-policy",
            args.frame_policy,
            "--update-every",
            str(args.update_every),
            "--video-key",
            args.video_key,
            "--output-dir",
            str(episode_report_dir),
            "--max-images",
            "0",
            "--no-contact-sheet",
            "--render-video",
            "--video-output",
            str(output_path),
            "--codec",
            args.codec,
            "--coord-range",
            str(args.coord_range),
            "--loc-range",
            str(args.loc_range),
            "--marker-radius",
            str(args.marker_radius),
            "--line-width",
            str(args.line_width),
        ]
        if args.max_video_frames is not None:
            command.extend(["--max-video-frames", str(args.max_video_frames)])
        if args.show_prompt_text:
            command.append("--show-prompt-text")
        commands.append(command)

    print(f"Rendering {len(commands)} prompt-point videos to {output_dir} (skipped existing: {skipped})")
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
        "dataset_path": str(dataset_path),
        "cot_json": str(cot_json),
        "output_dir": str(output_dir),
        "prompt_source": args.prompt_source,
        "traj_column": args.traj_column,
        "frame_policy": args.frame_policy,
        "update_every": int(args.update_every),
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
