#!/usr/bin/env python3
"""Conservative CPU continuation for a failed active QC200 Colab run.

Stage beside qc200-training (or pass --source-dir), then run with the active
config, captured pipeline log, and the original runtime's absolute UTC deadline.
This watcher never creates a run or edits frozen training sources.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import filecmp
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import time
import traceback


PHASE = "pipeline phase=fresh_normalization_then_B1_training"
GPU_LOG_MARKERS = (
    "cuda selected but no cuda device is available",
    "no cuda gpus are available",
    "cuda driver initialization failed",
    "cuda driver error",
    "cuda error: initialization error",
    "cuda driver version is insufficient",
)
TERMINAL_RECOVERY_STATES = {
    "triggered", "initial_sync_failed", "running_cpu", "completed_cpu",
    "paused_cpu", "cpu_failed", "ineligible_failure", "deadline_expired",
}
DRIVE_FOLDER_MIME = "application/vnd.google-apps.folder"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def process_records():
    """Return live, non-zombie Linux processes with argv and parent PID."""
    proc = Path("/proc")
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes().split(b"\0")
            argv = [part.decode(errors="replace") for part in raw if part]
            stat = (entry / "stat").read_text()
            tail = stat[stat.rfind(")") + 2:].split()
            state, ppid = tail[0], int(tail[1])
            if state not in ("Z", "X"):
                yield int(entry.name), ppid, argv
        except (OSError, ValueError, IndexError):
            continue


def matches_script(argv: list[str], script: str, config_path: Path) -> bool:
    names = {Path(arg).name for arg in argv}
    if script not in names:
        return False
    canonical = str(config_path.resolve())
    for i, arg in enumerate(argv[:-1]):
        if arg == "--config":
            value = argv[i + 1]
            try:
                if str(Path(value).resolve()) == canonical:
                    return True
            except OSError:
                pass
    return canonical in argv or str(config_path) in argv


def active_original_processes(config_path: Path) -> list[tuple[int, str]]:
    found = []
    for pid, _ppid, argv in process_records():
        for script in ("execute_pipeline.py", "train_qc200.py", "trainer.py"):
            if matches_script(argv, script, config_path):
                found.append((pid, script))
                break
    return found


def training_failure_evidence(status: dict, log_path: Path) -> tuple[bool, str]:
    error = str(status.get("error", ""))
    if status.get("phase") != "failed" or not re.search(r"Training exited\s+[1-9]", error, re.I):
        return False, "pipeline status does not identify a nonzero training child exit"
    if not log_path.is_file():
        return False, f"pipeline log not found: {log_path}"
    log = log_path.read_text(encoding="utf-8", errors="replace")
    phase_at = log.rfind(PHASE)
    if phase_at < 0:
        return False, "log has no training-phase marker"
    training_log = log[phase_at:]
    # The trainer traceback is followed by execute_pipeline's wrapper traceback;
    # inspect from the first one so the wrapper cannot hide the CUDA cause.
    traceback_at = training_log.find("Traceback (most recent call last):")
    if traceback_at < 0:
        return False, "training-phase log has no exception traceback"
    failure = training_log[traceback_at:].lower()
    marker = next((item for item in GPU_LOG_MARKERS if item in failure), None)
    if not marker:
        return False, "traceback lacks an explicit CUDA device/driver availability error"
    return True, f"training traceback contains: {marker}"


def cuda_unusable() -> tuple[bool, str]:
    """Confirm CUDA is unavailable or its device/driver cannot initialize."""
    try:
        import torch
    except Exception as exc:
        return False, f"PyTorch CUDA probe unavailable: {type(exc).__name__}: {exc}"
    try:
        if not torch.cuda.is_available():
            return True, "torch.cuda.is_available() is false"
        torch.cuda.init()
        probe = torch.empty((1,), device="cuda")
        probe.fill_(1)
        torch.cuda.synchronize()
        return False, "CUDA allocation and synchronization succeeded"
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        lower = message.lower()
        explicit = ("driver" in lower or "initialization error" in lower or
                    "no cuda device" in lower or "no cuda gpus" in lower or
                    "system has unsupported display driver" in lower)
        return (True, message) if explicit else (False, f"CUDA probe failed without a device/driver diagnosis: {message}")


def check_pause_guard(control: Path, pause_file: Path) -> bool:
    state_path = control / "persistence-state.json"
    try:
        state = read_json(state_path)
        synced = datetime.fromisoformat(state["utc"].replace("Z", "+00:00"))
        stale = (utcnow() - synced.astimezone(timezone.utc)).total_seconds() > 300
    except Exception:
        stale = True
    if stale and not pause_file.exists():
        pause_file.parent.mkdir(parents=True, exist_ok=True)
        pause_file.write_text("Persistence has not completed a verified pass for 300 seconds\n", encoding="utf-8")
    return stale


class DriveMirror:
    """Independent, byte-verified Drive persistence rooted in this existing run."""
    def __init__(self, source_dir: Path, run_folder_id: str, run_id: str,
                 approved_parent_id: str, max_file_bytes: int):
        sys.path.insert(0, str(source_dir))
        from drive_io import drive_client, find_or_create_folder, list_children, make_stable_snapshot, signature, upload_replace
        from googleapiclient.http import MediaIoBaseDownload
        self.drive_client = drive_client
        self.find_or_create_folder = find_or_create_folder
        self.list_children = list_children
        self.make_stable_snapshot = make_stable_snapshot
        self.signature = signature
        self.upload_replace = upload_replace
        self.MediaIoBaseDownload = MediaIoBaseDownload
        self.service = drive_client()
        meta = self.service.files().get(fileId=run_folder_id, fields="id,name,mimeType,trashed,parents,capabilities(canAddChildren)").execute()
        if meta.get("trashed") or meta.get("mimeType") != DRIVE_FOLDER_MIME or not meta.get("capabilities", {}).get("canAddChildren"):
            raise RuntimeError("pipeline-status Drive run folder is not writable")
        if meta.get("name") != run_id or approved_parent_id not in meta.get("parents", []):
            raise RuntimeError("pipeline-status Drive folder does not match this run and approved parent")
        self.root_id = run_folder_id
        self.max_file_bytes = max_file_bytes
        self.remote_dirs = {".": run_folder_id}
        self.previous = {}
        self.temp = Path(tempfile.mkdtemp(prefix="qc200-cpu-sync-"))
        self.lock = threading.Lock()

    def persist(self, path: Path, relative: Path):
        size = path.stat().st_size
        if size > self.max_file_bytes:
            raise IOError(f"Readback cap exceeded for {relative}: {size} > {self.max_file_bytes} bytes")
        sig = self.signature(path)
        key = str(relative)
        if self.previous.get(key) == sig:
            return None
        snap = self.make_stable_snapshot(path, self.temp)
        if snap is None:
            return None
        copy, original = snap
        folder = self.root_id
        parent = Path()
        for part in relative.parent.parts:
            parent = parent / part
            folder_key = parent.as_posix()
            if folder_key not in self.remote_dirs:
                self.remote_dirs[folder_key] = self.find_or_create_folder(self.service, folder, part)
            folder = self.remote_dirs[folder_key]
        file_id = self.upload_replace(self.service, folder, copy, remote_name=relative.name)
        received = self.temp / "readback.bin"
        with received.open("wb") as stream:
            transfer = self.MediaIoBaseDownload(stream, self.service.files().get_media(fileId=file_id), chunksize=8 * 1024**2)
            done = False
            while not done:
                _, done = transfer.next_chunk(num_retries=2)
                if stream.tell() > self.max_file_bytes:
                    raise IOError(f"Drive readback cap exceeded for {relative}")
        if not filecmp.cmp(copy, received, shallow=False):
            raise IOError(f"Drive byte readback differs for {relative}")
        if self.signature(path) == original:
            self.previous[key] = original
        return {"relative": key, "bytes": original[1], "file_id": file_id}

    def sync(self, control: Path, output: Path):
        with self.lock:
            verified = []
            for directory, prefix in ((control, Path("control")), (output, Path())):
                for path in sorted(directory.rglob("*")):
                    if not path.is_file() or path.is_symlink() or path.suffix in (".tmp", ".part"):
                        continue
                    result = self.persist(path, prefix / path.relative_to(directory))
                    if result:
                        verified.append(result)
            state_path = control / "persistence-state.json"
            state = {"utc": iso(utcnow()), "state": "verified_readback",
                     "verified_files_this_pass": verified,
                     "total_currently_persisted_files": len(self.previous),
                     "remote_run_folder_id": self.root_id}
            write_json(state_path, state)
            result = self.persist(state_path, Path("control") / state_path.name)
            if result:
                verified.append(result)
            return verified


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path, help="active pipeline config; read again at failure")
    ap.add_argument("--log", required=True, type=Path, help="captured execute_pipeline stdout/stderr log")
    ap.add_argument("--deadline-utc", required=True, help="absolute total runtime deadline, ISO-8601 UTC")
    ap.add_argument("--source-dir", type=Path, default=None, help="frozen qc200-training source directory")
    ap.add_argument("--poll-seconds", type=float, default=10)
    ap.add_argument("--final-sync-reserve-minutes", type=float, default=15)
    ap.add_argument("--readback-limit-bytes", type=int, default=536870912, help="max bytes per Drive file readback")
    args = ap.parse_args()
    config_path = args.config.resolve()
    source_dir = (args.source_dir or (Path(__file__).resolve().parent.parent / "qc200-training")).resolve()
    if not source_dir.is_dir():
        ap.error(f"source directory does not exist: {source_dir}")
    try:
        parsed_deadline = datetime.fromisoformat(args.deadline_utc.replace("Z", "+00:00"))
        if parsed_deadline.tzinfo is None:
            raise ValueError("timezone required")
        deadline = parsed_deadline.astimezone(timezone.utc)
    except ValueError:
        ap.error("--deadline-utc must be an ISO-8601 timestamp with timezone")
    if args.poll_seconds <= 0 or args.final_sync_reserve_minutes < 1 or args.readback_limit_bytes < 1:
        ap.error("poll, sync reserve, and readback limit must be positive")

    config = read_json(config_path)
    control_value = config.get("pipeline_control_dir")
    if not control_value:
        ap.error("active config has no pipeline_control_dir")
    control = Path(control_value).resolve()
    control.mkdir(parents=True, exist_ok=True)
    output = Path(config["output_dir"]).resolve()
    # Honor the runtime's recorded start time and configured total ceiling even
    # if a later watcher invocation is given a more generous CLI deadline.
    process_record = control / "pipeline-process.json"
    if process_record.is_file():
        try:
            started = datetime.fromisoformat(read_json(process_record)["started_utc"].replace("Z", "+00:00"))
            if started.tzinfo is not None:
                hard_deadline = started.astimezone(timezone.utc) + timedelta(
                    hours=float(config.get("max_runtime_hours", 10)))
                deadline = min(deadline, hard_deadline)
        except (KeyError, ValueError, TypeError):
            pass
    state_path = control / "cpu-recovery-status.json"
    budget_path = control / "cpu-recovery-deadline.json"
    lock_path = control / "cpu-recovery.lock"
    lock_stream = lock_path.open("a+")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Another CPU recovery watcher already holds the single-instance lock.", flush=True)
        return 2

    if state_path.is_file():
        old_state = read_json(state_path)
        if old_state.get("state") in TERMINAL_RECOVERY_STATES:
            print(f"Recovery is already terminal: {old_state['state']}", flush=True)
            return 0
    if budget_path.is_file():
        deadline = min(deadline, datetime.fromisoformat(read_json(budget_path)["deadline_utc"]).astimezone(timezone.utc))
    else:
        write_json(budget_path, {"created_utc": iso(utcnow()), "deadline_utc": iso(deadline), "source": "operator absolute deadline"})

    def state(name: str, **extra):
        body = {"utc": iso(utcnow()), "state": name, "run_id": config.get("run_id"), **extra}
        write_json(state_path, body)
        print(f"CPU recovery state={name}", flush=True)

    state("watching", deadline_utc=iso(deadline))
    while True:
        if utcnow() >= deadline:
            state("deadline_expired", deadline_utc=iso(deadline))
            return 0
        try:
            config = read_json(config_path)  # pipeline writes provenance/pause_file after audit
            pipeline_status_path = control / "pipeline-status.json"
            status = read_json(pipeline_status_path)
        except Exception as exc:
            state("watching", detail=f"waiting for pipeline metadata: {type(exc).__name__}: {exc}", deadline_utc=iso(deadline))
            time.sleep(args.poll_seconds)
            continue
        if status.get("phase") in ("B1_finished_train_dev_only", "B1_paused"):
            state("ineligible_failure", reason=f"original pipeline ended normally: {status.get('phase')}")
            return 0
        if status.get("phase") != "failed":
            time.sleep(args.poll_seconds)
            continue
        if status.get("run_id") != config.get("run_id"):
            state("ineligible_failure", reason="pipeline-status run_id differs from active config")
            return 0
        eligible, evidence = training_failure_evidence(status, args.log.resolve())
        if not eligible:
            state("ineligible_failure", reason=evidence)
            return 0
        processes = active_original_processes(config_path)
        if processes:
            time.sleep(args.poll_seconds)
            continue
        output = Path(config["output_dir"]).resolve()
        run_status_path = output / "run-status.json"
        if run_status_path.is_file():
            run_state = read_json(run_status_path).get("state")
            if run_state in ("complete_train_dev", "paused_runtime"):
                state("ineligible_failure", reason=f"trainer status is {run_state}")
                return 0
        checkpoint = output / "last.pt"
        if checkpoint.is_symlink() or not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
            state("ineligible_failure", reason="same-run output last.pt is missing; recovery never starts from scratch")
            return 0
        pause_value = config.get("pause_file")
        if not pause_value:
            state("ineligible_failure", reason="active config has no existing pause_file persistence guard")
            return 0
        pause_file = Path(pause_value).resolve()
        if pause_file.exists():
            state("ineligible_failure", reason="existing persistence pause marker is already set")
            return 0
        cuda_down, cuda_detail = cuda_unusable()
        if not cuda_down:
            state("ineligible_failure", reason=f"CUDA probe does not confirm device/driver unavailable: {cuda_detail}")
            return 0
        state("triggered", log_evidence=evidence, cuda_probe=cuda_detail, checkpoint=str(checkpoint))
        break

    # Use only existing run identity and Drive folder; no run folder is created.
    status = read_json(control / "pipeline-status.json")
    remote_id = status.get("drive_run_folder_id")
    if not remote_id:
        state("initial_sync_failed", error="pipeline-status.json lacks drive_run_folder_id")
        return 1
    try:
        mirror = DriveMirror(source_dir, remote_id, config["run_id"],
                             config.get("drive_parent_folder_id", ""), args.readback_limit_bytes)
        initial = mirror.sync(control, output)
        if not any(item.get("relative") == "last.pt" for item in initial) and str(Path("last.pt")) not in mirror.previous:
            raise IOError("Initial Drive sync did not verify output last.pt")
    except Exception as exc:
        write_json(control / "persistence-error.json", {"utc": iso(utcnow()), "error": f"initial sync: {type(exc).__name__}: {exc}"})
        state("initial_sync_failed", error=f"{type(exc).__name__}: {exc}")
        return 1

    remaining_hours = min(float(config.get("max_runtime_hours", 10)), (deadline - utcnow()).total_seconds() / 3600)
    cpu_hours = remaining_hours - args.final_sync_reserve_minutes / 60
    if cpu_hours <= max(0.02, float(config.get("save_reserve_minutes", 10)) / 60):
        state("deadline_expired", reason="insufficient remaining time for CPU checkpointing and final Drive sync", deadline_utc=iso(deadline))
        return 0
    cpu_config = dict(config)
    cpu_config["device"] = "cpu"
    cpu_config["resume"] = str(checkpoint)
    cpu_config["max_runtime_hours"] = cpu_hours
    cpu_config_path = control / "cpu-recovery-config.json"
    write_json(cpu_config_path, cpu_config)
    # Initial sync above must include the exact recovery config before launch.
    try:
        mirror.sync(control, output)
    except Exception as exc:
        write_json(control / "persistence-error.json", {"utc": iso(utcnow()), "error": f"recovery config sync: {type(exc).__name__}: {exc}"})
        state("initial_sync_failed", error=f"recovery config sync: {type(exc).__name__}: {exc}")
        return 1

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""  # trainer resume must not restore a broken CUDA RNG state
    env["PYTHONUNBUFFERED"] = "1"
    log_stream = (control / "cpu-recovery-training.log").open("a", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, str(source_dir / "train_qc200.py"), "--config", str(cpu_config_path)],
                            cwd=str(source_dir), env=env, stdout=log_stream, stderr=subprocess.STDOUT)
    state("running_cpu", cpu_pid=proc.pid, cpu_config=str(cpu_config_path), checkpoint=str(checkpoint),
          cpu_runtime_hours=cpu_hours, deadline_utc=iso(deadline), log_evidence=evidence, cuda_probe=cuda_detail)
    sync_error = {"value": None}
    sync_stop = threading.Event()

    def persistence_loop():
        while not sync_stop.wait(45):
            try:
                mirror.sync(control, output)
                sync_error["value"] = None
            except Exception as exc:
                sync_error["value"] = f"{type(exc).__name__}: {exc}"
                write_json(control / "persistence-error.json", {"utc": iso(utcnow()), "error": sync_error["value"]})

    sync_thread = threading.Thread(target=persistence_loop, name="cpu-recovery-drive-sync", daemon=True)
    sync_thread.start()
    while proc.poll() is None:
        check_pause_guard(control, pause_file)
        time.sleep(min(args.poll_seconds, 5))
    sync_stop.set()
    sync_thread.join()
    log_stream.close()
    cpu_status = read_json(output / "run-status.json") if (output / "run-status.json").is_file() else {}
    if proc.returncode == 0 and cpu_status.get("state") == "complete_train_dev":
        final_state = "completed_cpu"
    elif proc.returncode == 0 and cpu_status.get("state") == "paused_runtime":
        final_state = "paused_cpu"
    else:
        final_state = "cpu_failed"
    state(final_state, returncode=proc.returncode, training_state=cpu_status.get("state"),
          persistence_error=sync_error["value"])
    final_error = None
    try:
        mirror.sync(control, output)
    except Exception as exc:
        final_error = f"{type(exc).__name__}: {exc}"
        write_json(control / "persistence-error.json", {"utc": iso(utcnow()), "error": f"final sync: {final_error}"})
    if final_error:
        state(final_state, returncode=proc.returncode, training_state=cpu_status.get("state"),
              persistence_error=sync_error["value"], final_sync_error=final_error)
    return 0 if proc.returncode == 0 and not final_error else 1


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception:
        traceback.print_exc()
        raise
    raise SystemExit(exit_code)
