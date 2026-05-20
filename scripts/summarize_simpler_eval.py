#!/usr/bin/env python3
"""Summarize pi0.5 Bridge SimplerEnv evaluation logs."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re


SUMMARY_RE = re.compile(r"^([A-Za-z0-9_]+):\s+(\d+)/(\d+)\s+=\s+([0-9.]+)")
EPISODE_RE = re.compile(r"^\[Episode\s+\d+\]\s+(success|failure);")


def iter_logs(paths: list[Path]) -> list[Path]:
    logs: list[Path] = []
    for path in paths:
        if path.is_file():
            logs.append(path)
        elif path.is_dir():
            logs.extend(sorted(path.rglob("simpler_eval.log")))
    return sorted(dict.fromkeys(logs))


def parse_log(path: Path) -> list[dict[str, str | int | float]]:
    image_preprocess = "legacy_none"
    task_keys: list[str] = []
    summary_rows: list[dict[str, str | int | float]] = []
    episode_counts: Counter[str] = Counter()
    finished = False

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("Image preprocess:"):
            image_preprocess = line.split(":", 1)[1].strip()
        elif line.startswith("Tasks:"):
            task_keys = re.findall(r"'([^']+)'", line)
        elif line == "=== Summary ===":
            finished = True
        elif match := SUMMARY_RE.match(line):
            task, success, total, rate = match.groups()
            if task != "overall":
                summary_rows.append(
                    {
                        "run": path.parent.name,
                        "task": task,
                        "image_preprocess": image_preprocess,
                        "success": int(success),
                        "total": int(total),
                        "rate": float(rate),
                        "status": "complete" if finished else "summary",
                        "log": str(path),
                    }
                )
        elif match := EPISODE_RE.match(line):
            episode_counts[match.group(1)] += 1

    if summary_rows:
        return summary_rows

    total = episode_counts["success"] + episode_counts["failure"]
    if total == 0:
        return [
            {
                "run": path.parent.name,
                "task": task_keys[0] if len(task_keys) == 1 else "unknown",
                "image_preprocess": image_preprocess,
                "success": 0,
                "total": 0,
                "rate": 0.0,
                "status": "no_episodes",
                "log": str(path),
            }
        ]

    return [
        {
            "run": path.parent.name,
            "task": task_keys[0] if len(task_keys) == 1 else "unknown",
            "image_preprocess": image_preprocess,
            "success": episode_counts["success"],
            "total": total,
            "rate": episode_counts["success"] / total,
            "status": "incomplete",
            "log": str(path),
        }
    ]


def print_table(rows: list[dict[str, str | int | float]]) -> None:
    headers = ["run", "task", "image_preprocess", "success", "total", "rate", "status"]
    widths = {header: len(header) for header in headers}
    rendered_rows: list[dict[str, str]] = []
    for row in rows:
        rendered = {
            "run": str(row["run"]),
            "task": str(row["task"]),
            "image_preprocess": str(row["image_preprocess"]),
            "success": str(row["success"]),
            "total": str(row["total"]),
            "rate": f"{float(row['rate']):.4f}",
            "status": str(row["status"]),
        }
        rendered_rows.append(rendered)
        for header, value in rendered.items():
            widths[header] = max(widths[header], len(value))

    print("  ".join(header.ljust(widths[header]) for header in headers))
    print("  ".join("-" * widths[header] for header in headers))
    for row in rendered_rows:
        print("  ".join(row[header].ljust(widths[header]) for header in headers))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path("logs/simpler_eval/pi05_bridge_reproduce_30000")],
        help="Log files or directories to scan recursively.",
    )
    args = parser.parse_args()

    rows: list[dict[str, str | int | float]] = []
    for log_path in iter_logs(args.paths):
        rows.extend(parse_log(log_path))

    if not rows:
        raise SystemExit("No simpler_eval.log files found.")
    print_table(rows)


if __name__ == "__main__":
    main()
