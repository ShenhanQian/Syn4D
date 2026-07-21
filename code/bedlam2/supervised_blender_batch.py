from __future__ import annotations

import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Callable


ACTIVE_PROCESSES: dict[int, subprocess.Popen] = {}
ACTIVE_LOCK = threading.Lock()
STOP_REQUESTED = threading.Event()


@dataclass(frozen=True)
class ConversionTask:
    input_path: Path
    output_path: Path
    log_path: Path


@dataclass(frozen=True)
class ConversionResult:
    status: str
    input_path: Path
    output_path: Path
    log_path: Path
    returncode: int | None
    elapsed_seconds: float
    attempt: int


def terminate_process_tree(pid: int) -> None:
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            try:
                os.killpg(pid, signal.SIGTERM)
                time.sleep(5)
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                return
    except ProcessLookupError:
        return
    except Exception as exc:
        print(f"WARNING: failed to terminate process tree {pid}: {exc}", file=sys.stderr)


def terminate_active_processes() -> None:
    with ACTIVE_LOCK:
        pids = list(ACTIVE_PROCESSES)
    for pid in pids:
        terminate_process_tree(pid)


def register_process(proc: subprocess.Popen) -> None:
    with ACTIVE_LOCK:
        ACTIVE_PROCESSES[proc.pid] = proc


def unregister_process(proc: subprocess.Popen) -> None:
    with ACTIVE_LOCK:
        ACTIVE_PROCESSES.pop(proc.pid, None)


def output_looks_exported(task: ConversionTask, success_markers: tuple[str, ...]) -> bool:
    if not task.output_path.exists() or task.output_path.stat().st_size <= 0:
        return False
    if not task.log_path.exists():
        return False
    try:
        text = task.log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return all(marker in text for marker in success_markers)


def run_one_attempt(
    task: ConversionTask,
    command: list[str],
    timeout_seconds: int,
    stop_file: Path,
    success_markers: tuple[str, ...],
    attempt: int,
) -> ConversionResult:
    if STOP_REQUESTED.is_set() or stop_file.exists():
        STOP_REQUESTED.set()
        return ConversionResult("stopped", task.input_path, task.output_path, task.log_path, None, 0.0, attempt)

    task.output_path.parent.mkdir(parents=True, exist_ok=True)
    task.log_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    with task.log_path.open("a", encoding="utf-8", errors="replace") as log_file:
        log_file.write(f"\n=== attempt {attempt} ===\n")
        log_file.write(" ".join(command) + "\n")
        log_file.flush()
        kwargs = {"stdout": log_file, "stderr": subprocess.STDOUT}
        if os.name != "nt":
            kwargs["start_new_session"] = True
        proc = subprocess.Popen(command, **kwargs)
        register_process(proc)
        try:
            while True:
                returncode = proc.poll()
                if returncode is not None:
                    elapsed = time.perf_counter() - start_time
                    status = "ok" if returncode == 0 else "failed"
                    return ConversionResult(status, task.input_path, task.output_path, task.log_path, returncode, elapsed, attempt)
                if STOP_REQUESTED.is_set() or stop_file.exists():
                    STOP_REQUESTED.set()
                    terminate_process_tree(proc.pid)
                    elapsed = time.perf_counter() - start_time
                    return ConversionResult("stopped", task.input_path, task.output_path, task.log_path, None, elapsed, attempt)
                if timeout_seconds and (time.perf_counter() - start_time) > timeout_seconds:
                    terminate_process_tree(proc.pid)
                    elapsed = time.perf_counter() - start_time
                    if output_looks_exported(task, success_markers):
                        return ConversionResult("timeout_after_export", task.input_path, task.output_path, task.log_path, None, elapsed, attempt)
                    return ConversionResult("timeout", task.input_path, task.output_path, task.log_path, None, elapsed, attempt)
                time.sleep(1.0)
        finally:
            unregister_process(proc)


def run_task(
    task: ConversionTask,
    command_builder: Callable[[ConversionTask], list[str]],
    timeout_seconds: int,
    retries: int,
    stop_file: Path,
    success_markers: tuple[str, ...],
) -> ConversionResult:
    last_result: ConversionResult | None = None
    for attempt in range(1, retries + 2):
        last_result = run_one_attempt(
            task,
            command_builder(task),
            timeout_seconds,
            stop_file,
            success_markers,
            attempt,
        )
        if last_result.status in {"ok", "stopped", "timeout_after_export"}:
            return last_result
    if last_result is None:
        raise RuntimeError("unreachable: task produced no result")
    return last_result


def write_result(writer: csv.writer, result: ConversionResult) -> None:
    writer.writerow(
        [
            result.status,
            f"{result.elapsed_seconds:.1f}",
            result.attempt,
            result.returncode if result.returncode is not None else "",
            result.input_path,
            result.output_path,
            result.log_path,
        ]
    )


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:d}:{seconds:02d}"


def run_supervised_batch(
    *,
    tasks: list[ConversionTask],
    output_dir: Path,
    processes: int,
    timeout_seconds: int,
    retries: int,
    stop_file: Path,
    command_builder: Callable[[ConversionTask], list[str]],
    success_markers: tuple[str, ...],
    input_label: str,
) -> int:
    STOP_REQUESTED.clear()
    report_path = output_dir / "conversion_report.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {input_label} files to convert: {len(tasks)}")
    print(f"Starting {processes} Blender processes")
    print(f"Timeout per file: {timeout_seconds}s")
    print(f"Stop file: {stop_file}")
    print(f"Report: {report_path}")
    print("Per-file logs: " + str(output_dir / "_logs"))

    start_time = time.perf_counter()
    counts = {"ok": 0, "failed": 0, "timeout": 0, "timeout_after_export": 0, "stopped": 0}
    with report_path.open("w", newline="", encoding="utf-8") as report_file:
        writer = csv.writer(report_file)
        writer.writerow(["status", "elapsed_seconds", "attempt", "returncode", "input", "output", "log"])
        try:
            with ThreadPoolExecutor(max_workers=processes) as executor:
                futures = [
                    executor.submit(
                        run_task,
                        task,
                        command_builder,
                        timeout_seconds,
                        retries,
                        stop_file,
                        success_markers,
                    )
                    for task in tasks
                ]
                for future in as_completed(futures):
                    result = future.result()
                    counts[result.status] = counts.get(result.status, 0) + 1
                    write_result(writer, result)
                    report_file.flush()
                    done = sum(counts.values())
                    elapsed = time.perf_counter() - start_time
                    rate = done / elapsed if elapsed > 0 else 0.0
                    remaining = len(tasks) - done
                    eta = remaining / rate if rate > 0 else 0.0
                    print(
                        f"Progress: {done}/{len(tasks)} "
                        f"ok={counts['ok']} failed={counts['failed']} "
                        f"timeout={counts['timeout']} "
                        f"timeout_after_export={counts['timeout_after_export']} "
                        f"stopped={counts['stopped']} "
                        f"last={result.status}:{format_duration(result.elapsed_seconds)} "
                        f"elapsed={format_duration(elapsed)} eta={format_duration(eta)} "
                        f"rate={rate * 60:.2f}/min",
                        flush=True,
                    )
                    if result.status == "stopped":
                        STOP_REQUESTED.set()
                        terminate_active_processes()
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
        except KeyboardInterrupt:
            print("Interrupted. Killing active Blender children...", file=sys.stderr)
            STOP_REQUESTED.set()
            terminate_active_processes()
            return 130
        except Exception:
            print("Error. Killing active Blender children...", file=sys.stderr)
            STOP_REQUESTED.set()
            terminate_active_processes()
            raise
        finally:
            terminate_active_processes()

    elapsed = time.perf_counter() - start_time
    print(f"Finished in {elapsed:.1f}s")
    print(f"Summary: {counts}")
    if counts["failed"] or counts["timeout"]:
        return 1
    if counts["stopped"]:
        return 130
    return 0
