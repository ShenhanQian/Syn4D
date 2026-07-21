#!/usr/bin/env python
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (c) 2025 Max Planck Society
#
# Batch convert BEDLAM2 SMPL-X .npz animations to Alembic .abc files.

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from supervised_blender_batch import ConversionTask, run_supervised_batch


BLENDER_APP_PATH = r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"
DEFAULT_PROCESSES = 4
DEFAULT_TIMEOUT_SECONDS = 300
SUCCESS_LOG_MARKERS = ("Exported:", "Finished. Time:")
WORKER_SCRIPT = Path(__file__).with_name("smplx_anim_to_alembic.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch convert BEDLAM2 SMPL-X .npz animations to Alembic .abc files.",
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing BEDLAM2 source .npz files.")
    parser.add_argument("output_dir", type=Path, help="Directory where .abc files and logs will be written.")
    parser.add_argument("processes", nargs="?", type=int, default=DEFAULT_PROCESSES)
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Kill a Blender child if one file takes longer than this. Use 0 to disable.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=0,
        help="Number of retries after a failed or timed-out conversion.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip output .abc files that already exist and have nonzero size.",
    )
    parser.add_argument(
        "--stop-file",
        type=Path,
        default=None,
        help="If this file appears, active Blender children are killed and queued work is stopped.",
    )
    parser.add_argument("--blender", default=BLENDER_APP_PATH, help="Path to the Blender executable.")
    args = parser.parse_args()
    if args.processes <= 0:
        parser.error("processes must be a positive integer")
    if args.timeout_seconds < 0:
        parser.error("--timeout-seconds must be >= 0")
    if args.retries < 0:
        parser.error("--retries must be >= 0")
    if args.stop_file is None:
        args.stop_file = args.output_dir / "STOP"
    return args


def output_path_for(input_path: Path, output_dir: Path) -> Path:
    if input_path.name.startswith("moyo"):
        return output_dir / "moyo" / f"{input_path.stem}.abc"
    body_name = input_path.name.rsplit("_", maxsplit=1)[0]
    return output_dir / body_name / f"{input_path.stem}.abc"


def log_path_for(output_path: Path, output_dir: Path) -> Path:
    return output_dir / "_logs" / output_path.parent.name / f"{output_path.stem}.log"


def build_tasks(input_dir: Path, output_dir: Path, skip_existing: bool) -> list[ConversionTask]:
    tasks: list[ConversionTask] = []
    for input_path in sorted(input_dir.rglob("*.npz")):
        output_path = output_path_for(input_path, output_dir)
        if skip_existing and output_path.exists() and output_path.stat().st_size > 0:
            continue
        tasks.append(ConversionTask(input_path, output_path, log_path_for(output_path, output_dir)))
    return tasks


def command_builder(blender_app_path: str):
    def build(task: ConversionTask) -> list[str]:
        return [
            blender_app_path,
            "--background",
            "--python",
            str(WORKER_SCRIPT),
            "--",
            "--input",
            str(task.input_path),
            "--output",
            str(task.output_path),
        ]

    return build


def main() -> int:
    args = parse_args()
    tasks = build_tasks(args.input_dir, args.output_dir, args.skip_existing)
    return run_supervised_batch(
        tasks=tasks,
        output_dir=args.output_dir,
        processes=args.processes,
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
        stop_file=args.stop_file,
        command_builder=command_builder(args.blender),
        success_markers=SUCCESS_LOG_MARKERS,
        input_label=".npz",
    )


if __name__ == "__main__":
    sys.exit(main())
