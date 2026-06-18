#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import aloha_postprocess as post


REPO_ROOT = Path(__file__).resolve().parents[2]
SCREEN_SCRIPT = REPO_ROOT / "tools" / "data_quality" / "screen_hdf5_episodes.py"
CONVERT_SCRIPT = REPO_ROOT / "tools" / "conversion" / "convert_hdf5_to_lerobot_absolute.py"
TRAJECTORY_SCRIPT = REPO_ROOT / "tools" / "trajectory" / "generate_aloha_fk_trajectory.py"
COT_SCRIPT = REPO_ROOT / "tools" / "trajectory" / "generate_aloha_bimanual_cot_prompts.py"
VISUALIZE_ALL_SCRIPT = REPO_ROOT / "tools" / "trajectory" / "visualize_all_aloha_fk_trajectories.py"


def split_tasks(values: list[str]) -> list[str]:
    tasks: list[str] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                tasks.append(part)
    deduped = []
    seen = set()
    for task in tasks:
        if task not in seen:
            deduped.append(task)
            seen.add(task)
    return deduped


def run_command(command: list[str], cwd: Path) -> None:
    print("\n[RUN] " + " ".join(command), flush=True)
    result = subprocess.run(command, cwd=str(cwd), check=False)
    if result.returncode != 0:
        raise SystemExit(f"Command failed with exit code {result.returncode}: {' '.join(command)}")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_task_dirs(raw_root: Path, tasks: list[str]) -> None:
    missing = []
    empty = []
    for task in tasks:
        task_dir = raw_root / task
        if not task_dir.is_dir():
            missing.append(str(task_dir))
            continue
        if not list(task_dir.glob("episode_*.hdf5")) and not list(task_dir.glob("episode-*.hdf5")):
            empty.append(str(task_dir))
    if missing or empty:
        message = []
        if missing:
            message.append("Missing task directories:\n" + "\n".join(f"  {path}" for path in missing))
        if empty:
            message.append("Task directories without episode_*.hdf5 files:\n" + "\n".join(f"  {path}" for path in empty))
        raise SystemExit("\n".join(message))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the generic Aloha HDF5 processing pipeline: screen task HDF5 files, "
            "convert valid episodes to LeRobot absolute format, then generate FK trajectories, CoT prompts, and videos."
        )
    )
    parser.add_argument("--tasks", nargs="+", required=True, help="Task folder names under --raw-root.")
    parser.add_argument("--dataset-name", required=True, help="Output folder name under --dataset-root.")
    parser.add_argument("--raw-root", type=Path, default=Path("~/data/aloha_pipeline"))
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "playground" / "Datasets")
    parser.add_argument("--robot-type", default="aloha_piper_absolute")
    parser.add_argument("--repo-id", default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild generated LeRobot outputs while keeping data_quality screening files.",
    )
    parser.add_argument("--screen-only", action="store_true", help="Only write HDF5 screening outputs under the dataset directory.")
    parser.add_argument("--skip-screen", action="store_true", help="Reuse an existing data_quality/hdf5_screen_report.json.")
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--skip-trajectory", action="store_true")
    parser.add_argument("--skip-cot", action="store_true")
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument("--skip-subtask-patch", action="store_true", help="Do not patch HDF5 subtask labels into parquets.")
    parser.add_argument("--skip-gripper-binary", action="store_true", help="Do not create the *_gripper_binary dataset.")
    parser.add_argument("--skip-reencode-videos", action="store_true", help="Do not re-encode videos in the final dataset.")
    parser.add_argument("--skip-subgoal", action="store_true", help="Do not generate cam_high_subgoal videos.")
    parser.add_argument("--skip-traj-cot-patch", action="store_true", help="Do not write traj_cot back into final parquets.")
    parser.add_argument("--num-workers", type=int, default=8, help="Workers for HDF5 screening.")
    parser.add_argument("--cot-workers", type=int, default=8)
    parser.add_argument("--video-workers", type=int, default=4)
    parser.add_argument("--active-arms", choices=["any", "left", "right", "both"], default="any")
    parser.add_argument("--min-frames", type=int, default=20)
    parser.add_argument("--min-joint-range", type=float, default=0.03)
    parser.add_argument("--min-cumulative-joint-movement", type=float, default=0.15)
    parser.add_argument("--check-images", action="store_true")
    parser.add_argument(
        "--manual-bad-episodes",
        nargs="*",
        default=None,
        help=(
            "Operator-specified bad HDF5 episodes added to automatic screening. "
            "Accepts paths, task:episode_id, task:start:stop, episode_id, or start:stop."
        ),
    )
    parser.add_argument("--manual-bad-list", type=Path, default=None, help="Text file of manual bad episode tokens.")
    parser.add_argument("--episode-index-mode", choices=["auto", "preserve", "sequential"], default="sequential")
    parser.add_argument("--limit", type=int, default=None, help="Debug option passed to conversion after screening.")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--traj-json-name", default="fk_bimanual.json")
    parser.add_argument("--cot-json-name", default="cot_text_prompts.json")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--left-gripper-dim", type=int, default=6)
    parser.add_argument("--right-gripper-dim", type=int, default=13)
    parser.add_argument("--gripper-binary-suffix", default="_gripper_binary")
    parser.add_argument("--gripper-close-threshold", type=float, default=0.05)
    parser.add_argument("--gripper-open-value", type=float, default=0.09)
    parser.add_argument("--gripper-close-value", type=float, default=0.0)
    parser.add_argument("--subgoal-window", type=int, default=60)
    parser.add_argument("--subgoal-camera", default="cam_high")
    parser.add_argument("--traj-cot-window", type=int, default=60)
    parser.add_argument("--traj-cot-column", default="traj_cot")
    parser.add_argument("--postprocess-video-codec", default="libsvtav1")
    parser.add_argument("--postprocess-video-crf", type=int, default=30)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    tasks = split_tasks(args.tasks)
    if not tasks:
        raise SystemExit("No task names were provided.")

    raw_root = args.raw_root.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    dataset_path = dataset_root / args.dataset_name
    data_quality_dir = dataset_path / "data_quality"
    report_path = data_quality_dir / "hdf5_screen_report.json"
    bad_list_path = data_quality_dir / "bad_hdf5_episodes.txt"
    repo_id = args.repo_id or f"local/{args.dataset_name}"
    validate_task_dirs(raw_root, tasks)

    if args.skip_screen:
        if not report_path.is_file():
            raise SystemExit(f"Missing existing screen report: {report_path}")
        if not bad_list_path.is_file():
            bad_list_path.parent.mkdir(parents=True, exist_ok=True)
            bad_list_path.write_text("", encoding="utf-8")
        print(f"[INFO] Reusing screen report: {report_path}")
    else:
        screen_command = [
            sys.executable,
            str(SCREEN_SCRIPT),
            "--root",
            str(raw_root),
            "--tasks",
            *tasks,
            "--output",
            str(report_path),
            "--bad-list",
            str(bad_list_path),
            "--num-workers",
            str(args.num_workers),
            "--active-arms",
            args.active_arms,
            "--min-frames",
            str(args.min_frames),
            "--min-joint-range",
            str(args.min_joint_range),
            "--min-cumulative-joint-movement",
            str(args.min_cumulative_joint_movement),
        ]
        if args.check_images:
            screen_command.append("--check-images")
        if args.manual_bad_episodes:
            screen_command.extend(["--manual-bad-episodes", *args.manual_bad_episodes])
        if args.manual_bad_list is not None:
            screen_command.extend(["--manual-bad-list", str(args.manual_bad_list.expanduser())])
        run_command(screen_command, REPO_ROOT)

    screen_data = read_json(report_path)
    print(
        f"[INFO] HDF5 screening: {screen_data['num_files']} files, "
        f"{screen_data['num_bad']} bad episodes."
    )

    if args.screen_only:
        write_json(
            data_quality_dir / "pipeline_request.json",
            {
                "raw_root": str(raw_root),
                "dataset_path": str(dataset_path),
                "tasks": tasks,
                "screen_only": True,
            },
        )
        print(f"[DONE] Screen report: {report_path}")
        print(f"[DONE] Bad list: {bad_list_path}")
        return

    if not args.skip_convert:
        convert_command = [
            sys.executable,
            str(CONVERT_SCRIPT),
            "--src-dir",
            str(raw_root),
            "--tasks",
            *tasks,
            "--out-dir",
            str(dataset_path),
            "--repo-id",
            repo_id,
            "--robot-type",
            args.robot_type,
            "--episode-index-mode",
            args.episode_index_mode,
            "--screen-report",
            str(report_path),
            "--ffmpeg-bin",
            args.ffmpeg_bin,
        ]
        if args.overwrite:
            convert_command.append("--overwrite")
        if args.limit is not None:
            convert_command.extend(["--limit", str(args.limit)])
        run_command(convert_command, REPO_ROOT)

    postprocess_report: dict[str, Any] = {}
    if not args.skip_subtask_patch:
        print("\n[POST] Patching subtask labels from HDF5 source files", flush=True)
        postprocess_report["subtask_patch"] = post.patch_subtasks_from_hdf5(dataset_path)

    traj_json = dataset_path / "trajectory_data" / args.traj_json_name
    cot_json = dataset_path / "trajectory_data" / args.cot_json_name

    if not args.skip_trajectory:
        run_command(
            [
                sys.executable,
                str(TRAJECTORY_SCRIPT),
                "--dataset-path",
                str(dataset_path),
                "--include-arms",
                "both",
                "--output",
                str(traj_json),
            ],
            REPO_ROOT,
        )

    if not args.skip_cot:
        run_command(
            [
                sys.executable,
                str(COT_SCRIPT),
                "--dataset_path",
                str(dataset_path),
                "--traj_json",
                str(traj_json),
                "--output_path",
                str(cot_json),
                "--image_size",
                str(args.image_size),
                "--left_gripper_dim",
                str(args.left_gripper_dim),
                "--right_gripper_dim",
                str(args.right_gripper_dim),
                "--num_workers",
                str(args.cot_workers),
            ],
            REPO_ROOT,
        )

    if not args.skip_videos:
        run_command(
            [
                sys.executable,
                str(VISUALIZE_ALL_SCRIPT),
                "--dataset-path",
                str(dataset_path),
                "--traj-json",
                str(traj_json),
                "--arm",
                "both",
                "--num-workers",
                str(args.video_workers),
            ],
            REPO_ROOT,
        )

    final_dataset_path = dataset_path
    if not args.skip_gripper_binary:
        final_dataset_path = dataset_root / f"{args.dataset_name}{args.gripper_binary_suffix}"
        print(f"\n[POST] Creating gripper-binary dataset: {final_dataset_path}", flush=True)
        postprocess_report["gripper_binary"] = post.create_gripper_binary_dataset(
            dataset_path,
            final_dataset_path,
            repo_id=f"{repo_id}{args.gripper_binary_suffix}",
            overwrite=args.overwrite,
            left_gripper_joint=args.left_gripper_dim,
            right_gripper_joint=args.right_gripper_dim,
            close_threshold=args.gripper_close_threshold,
            open_value=args.gripper_open_value,
            close_value=args.gripper_close_value,
        )

    final_cot_json = final_dataset_path / "trajectory_data" / args.cot_json_name

    if not args.skip_reencode_videos:
        print(f"\n[POST] Re-encoding final dataset videos: {final_dataset_path}", flush=True)
        postprocess_report["reencode_videos"] = post.reencode_videos(
            final_dataset_path,
            codec=args.postprocess_video_codec,
            crf=args.postprocess_video_crf,
        )

    if not args.skip_subgoal:
        print(f"\n[POST] Generating subgoal videos: {final_dataset_path}", flush=True)
        postprocess_report["subgoal"] = post.add_subgoal_videos(
            final_dataset_path,
            camera=args.subgoal_camera,
            window=args.subgoal_window,
            codec=args.postprocess_video_codec,
            crf=args.postprocess_video_crf,
            overwrite=args.overwrite,
        )

    if not args.skip_traj_cot_patch:
        print(f"\n[POST] Writing {args.traj_cot_column} column: {final_dataset_path}", flush=True)
        postprocess_report["traj_cot_patch"] = post.patch_traj_cot_column(
            final_dataset_path,
            cot_json=final_cot_json,
            window=args.traj_cot_window,
            column=args.traj_cot_column,
        )

    summary = {
        "raw_root": str(raw_root),
        "dataset_path": str(dataset_path),
        "final_dataset_path": str(final_dataset_path),
        "dataset_name": args.dataset_name,
        "repo_id": repo_id,
        "robot_type": args.robot_type,
        "tasks": tasks,
        "screen_report": str(report_path),
        "bad_list": str(bad_list_path),
        "num_hdf5_files": screen_data["num_files"],
        "num_bad_hdf5_files": screen_data["num_bad"],
        "skipped_convert": bool(args.skip_convert),
        "skipped_trajectory": bool(args.skip_trajectory),
        "skipped_cot": bool(args.skip_cot),
        "skipped_videos": bool(args.skip_videos),
        "skipped_subtask_patch": bool(args.skip_subtask_patch),
        "skipped_gripper_binary": bool(args.skip_gripper_binary),
        "skipped_reencode_videos": bool(args.skip_reencode_videos),
        "skipped_subgoal": bool(args.skip_subgoal),
        "skipped_traj_cot_patch": bool(args.skip_traj_cot_patch),
        "trajectory_json": str(traj_json),
        "cot_json": str(cot_json),
        "final_cot_json": str(final_cot_json),
        "visualization_dir": str(dataset_path / "trajectory_data" / "visualizations_all"),
        "postprocess": postprocess_report,
    }
    write_json(data_quality_dir / "pipeline_summary.json", summary)
    print(f"[DONE] Dataset pipeline complete: {dataset_path}")
    print(f"[DONE] Screen report: {report_path}")


if __name__ == "__main__":
    main()
