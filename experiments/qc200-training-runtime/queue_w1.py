#!/usr/bin/env python3
"""Queue a fresh W1 run behind B1 and persist it in a separate Drive folder."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback


FINAL_SYNC_MINUTES = 15
MINIMUM_TRAINING_MINUTES = 30
POLL_SECONDS = 10


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_deadline(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("deadline must include a timezone")
    return parsed.astimezone(timezone.utc)


def select_device() -> tuple[str, str | None]:
    try:
        import torch
    except Exception as exc:
        return "cpu", f"PyTorch CUDA probe unavailable: {type(exc).__name__}: {exc}"
    if not torch.cuda.is_available():
        return "cpu", "CUDA is unavailable"
    try:
        torch.cuda.init()
        probe = torch.empty((1,), device="cuda")
        probe.fill_(1)
        torch.cuda.synchronize()
        return "cuda", None
    except Exception as exc:
        return "cpu", f"CUDA probe failed: {type(exc).__name__}: {exc}"


def active_b1_processes(config_path: Path, process_records, matches_script) -> list[tuple[int, str]]:
    active = []
    for pid, _ppid, argv in process_records():
        for script in ("execute_pipeline.py", "train_qc200.py", "trainer.py", "cpu_fallback_watch.py"):
            if matches_script(argv, script, config_path):
                active.append((pid, script))
                break
    return active


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--b1-config", required=True, type=Path)
    ap.add_argument("--w1-run-id", required=True)
    ap.add_argument("--deadline-utc", required=True)
    ap.add_argument("--source-dir", required=True, type=Path)
    ap.add_argument("--min-remaining-minutes", type=float, default=MINIMUM_TRAINING_MINUTES,
                    help="minimum W1 trainer window, excluding the final Drive sync reserve")
    ap.add_argument("--final-sync-minutes", type=float, default=FINAL_SYNC_MINUTES)
    ap.add_argument("--poll-seconds", type=float, default=POLL_SECONDS)
    args = ap.parse_args()

    b1_config_path = args.b1_config.resolve()
    source_dir = args.source_dir.resolve()
    if not source_dir.is_dir():
        ap.error(f"source directory does not exist: {source_dir}")
    try:
        deadline = parse_deadline(args.deadline_utc)
    except ValueError as exc:
        ap.error(str(exc))
    if args.min_remaining_minutes <= 0 or args.final_sync_minutes < 1 or args.poll_seconds <= 0:
        ap.error("minimum training window, final sync reserve, and poll interval must be positive")

    sys.path.insert(0, str(source_dir))
    runtime_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(runtime_dir))
    from run_contract import load_config, verify_feature_inventory, write_json as contract_write_json
    from preprocess_qc200 import read_markers
    from cpu_fallback_watch import DriveMirror, matches_script, process_records
    from drive_io import drive_client, list_children, find_or_create_folder

    b1_config = load_config(b1_config_path)
    if b1_config.get("variant", "B1") != "B1":
        ap.error("--b1-config must identify a B1 run")
    if args.min_remaining_minutes <= float(b1_config.get("save_reserve_minutes", 10)):
        ap.error("minimum W1 window must exceed the trainer's checkpoint reserve")
    if not args.w1_run_id or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.w1_run_id):
        ap.error("--w1-run-id must be a safe single directory name")
    if args.w1_run_id == b1_config["run_id"]:
        ap.error("W1 requires a run ID distinct from B1")
    control = Path(b1_config.get("pipeline_control_dir", "")).resolve()
    if not b1_config.get("pipeline_control_dir"):
        ap.error("B1 config has no pipeline_control_dir")
    control.mkdir(parents=True, exist_ok=True)
    lock_stream = (control / "w1-queue.lock").open("a+")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Another W1 queue process holds the single-instance lock.", flush=True)
        return 2

    queue_config_path = control / "w1-queue-config.json"
    queue_status_path = control / "w1-queue-status.json"
    w1_control = control / "w1-control"
    w1_control.mkdir(parents=True, exist_ok=True)
    identity = {
        "b1_config": str(b1_config_path), "b1_run_id": b1_config["run_id"],
        "w1_run_id": args.w1_run_id, "source_dir": str(source_dir),
        "deadline_utc": iso(deadline), "min_remaining_minutes": args.min_remaining_minutes,
        "final_sync_minutes": args.final_sync_minutes,
    }
    if queue_config_path.exists():
        saved = read_json(queue_config_path)
        if any(saved.get(key) != value for key, value in identity.items()):
            raise ValueError("Existing W1 queue identity differs; refusing to replace it")
    else:
        write_json(queue_config_path, identity)
    write_json(w1_control / "w1-queue-config.json", identity)

    status = read_json(queue_status_path) if queue_status_path.exists() else {}

    def publish(state: str, **extra) -> None:
        body = {"utc": iso(utcnow()), "state": state, **identity, **extra}
        write_json(queue_status_path, body)
        write_json(w1_control / "w1-queue-status.json", body)
        print(f"W1 queue state={state}", flush=True)

    terminal = {"complete_train_dev", "paused_runtime", "failed", "launch_uncertain", "initial_sync_failed"}
    if status.get("state") in terminal:
        print(f"W1 queue is already terminal: {status['state']}", flush=True)
        return 0
    if status.get("state") == "running":
        w1_config_path = w1_control / "w1-config.json"
        if w1_config_path.is_file() and active_b1_processes(w1_config_path, process_records, matches_script):
            print("W1 training is already active; refusing a duplicate.", flush=True)
            return 0
        output_value = status.get("output_dir")
        run_status_path = Path(output_value) / "run-status.json" if output_value else Path()
        saved_run = read_json(run_status_path) if run_status_path.is_file() else {}
        if saved_run.get("state") in ("complete_train_dev", "paused_runtime"):
            publish(saved_run["state"], recovered_status=True, training_state=saved_run["state"])
        else:
            publish("failed", reason="W1 was marked running but no matching trainer remains; auto-retry is disabled")
        return 2
    if status.get("state") == "launching":
        w1_config_path = w1_control / "w1-config.json"
        if w1_config_path.is_file() and active_b1_processes(w1_config_path, process_records, matches_script):
            print("W1 launch is already active; refusing a duplicate.", flush=True)
            return 0
        publish("launch_uncertain", reason="launch marker exists but no matching process; manual inspection required")
        return 2
    preserved = {key: status[key] for key in ("drive_run_folder_id", "output_dir") if key in status}
    publish("queued", reason="waiting for B1 train/dev completion and original persistence processes to exit", **preserved)

    while utcnow() < deadline:
        b1_config = load_config(b1_config_path)
        b1_output = Path(b1_config["output_dir"]).resolve()
        b1_control_status_path = control / "pipeline-status.json"
        b1_run_status_path = b1_output / "run-status.json"
        try:
            pipeline = read_json(b1_control_status_path)
            run_status = read_json(b1_run_status_path)
        except (OSError, ValueError):
            time.sleep(args.poll_seconds)
            continue
        recovery_path = control / "cpu-recovery-status.json"
        try:
            recovery = read_json(recovery_path) if recovery_path.is_file() else {}
        except (OSError, ValueError):
            recovery = {}
        if pipeline.get("run_id") != b1_config["run_id"]:
            time.sleep(args.poll_seconds)
            continue
        normal_complete = pipeline.get("phase") == "B1_finished_train_dev_only" and pipeline.get("training_state") == "complete_train_dev"
        cpu_complete = (recovery.get("state") == "completed_cpu" and recovery.get("training_state") == "complete_train_dev"
                        and recovery.get("run_id") == b1_config["run_id"])
        if run_status.get("state") == "complete_train_dev" and (normal_complete or cpu_complete):
            active = active_b1_processes(b1_config_path, process_records, matches_script)
            try:
                persistence = read_json(control / "persistence-state.json") if (control / "persistence-state.json").is_file() else {}
            except (OSError, ValueError):
                persistence = {}
            remote_id = pipeline.get("drive_run_folder_id")
            if not active and persistence.get("state") == "verified_readback" and persistence.get("remote_run_folder_id") == remote_id:
                break
        time.sleep(args.poll_seconds)
    else:
        publish("queued", reason="deadline reached before B1 completion/exit; no W1 launch")
        return 0

    active = active_b1_processes(b1_config_path, process_records, matches_script)
    if active:
        publish("queued", reason=f"waiting for active B1 processes: {active}")
        return 0
    remaining = (deadline - utcnow()).total_seconds()
    minimum_window = args.min_remaining_minutes * 60
    final_sync_reserve = args.final_sync_minutes * 60
    if remaining - final_sync_reserve < minimum_window:
        publish("queued", reason="insufficient remaining time for minimum W1 training window plus final Drive sync",
                remaining_seconds=max(0, remaining))
        return 0

    b1_config = load_config(b1_config_path)
    b1_output = Path(b1_config["output_dir"]).resolve()
    validation_source = b1_output / "preprocessing"
    report_path = validation_source / "feature_validation.json"
    if not report_path.is_file():
        raise FileNotFoundError("B1 full feature validation metadata is missing")
    report = read_json(report_path)
    if (not report.get("full_corpus") or report.get("missing") or
            report.get("feature_version") != b1_config["feature_version"] or
            not report.get("provenance_approved")):
        raise ValueError("B1 full-corpus feature validation is not approved")
    expected = b1_config.get("expected_partition_rows", {"train": 261865, "dev": 73387, "test": 38384})
    frozen = {"train": 261865, "dev": 73387, "test": 38384}
    synthetic = b1_config.get("cohort_kind") == "synthetic-test" and b1_config["feature_version"].startswith("synthetic-")
    if set(expected) != set(frozen) or (expected != frozen and not synthetic):
        raise ValueError("B1 validated partition counts do not match the supported cohort")
    if {key: report["splits"][key]["rows"] for key in expected} != expected:
        raise ValueError("B1 validation partition counts differ from the approved cohort")
    verify_feature_inventory(b1_config, validation_source)
    validated = read_json(validation_source / "validated_marker_rows.json")
    if validated != [list(row) for row in read_markers(Path(b1_config["marker_dir"]))]:
        raise ValueError("B1 markers differ from validated marker identities")

    output_dir = b1_output.parent / args.w1_run_id
    w1_config_path = w1_control / "w1-config.json"
    prepared_status = read_json(queue_status_path) if queue_status_path.is_file() else {}
    remote_id = prepared_status.get("drive_run_folder_id")
    if output_dir.exists():
        if not (output_dir / "preprocessing").is_dir() or any(p.name != "preprocessing" for p in output_dir.iterdir()):
            raise FileExistsError(f"W1 output is not a fresh queued output: {output_dir}")
        if not w1_config_path.is_file():
            raise ValueError("Existing W1 preparation has no saved identity config")
        w1_config = load_config(w1_config_path)
    else:
        output_dir.mkdir(parents=True)
        shutil.copytree(validation_source, output_dir / "preprocessing")
        if any(p.is_symlink() for p in (output_dir / "preprocessing").rglob("*")):
            raise ValueError("Validation metadata contains symlinks")
        device, device_reason = select_device()
        runtime_hours = min(float(b1_config.get("max_runtime_hours", 10)),
                            (deadline - utcnow()).total_seconds() / 3600 - args.final_sync_minutes / 60)
        w1_config = dict(b1_config)
        w1_config.update(run_id=args.w1_run_id, variant="W1", output_dir=str(output_dir),
                         pipeline_control_dir=str(w1_control), device=device,
                         max_runtime_hours=max(0.01, runtime_hours),
                         pause_file=str(w1_control / "pause-training.request"))
        w1_config.pop("resume", None)
        w1_config.pop("normalization", None)
        write_json(w1_config_path, w1_config)
        contract_write_json(w1_control / "w1-config.json", w1_config)
        write_json(w1_control / "w1-device-selection.json", {"device": device, "reason": device_reason})
        if runtime_hours * 60 < args.min_remaining_minutes:
            publish("queued", reason="insufficient remaining time after W1 preparation; no trainer launched",
                    output_dir=str(output_dir))
            return 0
    if w1_config.get("run_id") != args.w1_run_id or w1_config.get("variant") != "W1" or Path(w1_config["output_dir"]).resolve() != output_dir:
        raise ValueError("Saved W1 configuration does not match queued identity")
    shared_fields = ("feature_version", "preproc_dir", "marker_dir", "seed", "epochs", "batch_size",
                     "eval_batch_size", "workers", "lr", "weight_decay", "save_reserve_minutes",
                     "sample_rate_hz", "expected_partition_rows", "feature_contract_approved",
                     "legacy_provenance_record", "preprocess_mode", "drive_parent_folder_id")
    if any(w1_config.get(key) != b1_config.get(key) for key in shared_fields):
        raise ValueError("W1 source/features/markers/training settings differ from the completed B1 identity")
    if w1_config.get("resume") or (output_dir / "last.pt").exists() or (output_dir / "best.pt").exists():
        raise ValueError("W1 must start fresh and may not reuse B1 weights/checkpoints")

    if not remote_id:
        service = drive_client()
        parent = b1_config["drive_parent_folder_id"]
        parent_meta = service.files().get(fileId=parent, fields="id,name,mimeType,trashed,capabilities(canAddChildren)").execute()
        if (parent_meta.get("trashed") or parent_meta.get("mimeType") != "application/vnd.google-apps.folder" or
                not parent_meta.get("capabilities", {}).get("canAddChildren")):
            raise ValueError("Approved W1 Drive parent is not a writable folder")
        if list_children(service, parent, args.w1_run_id):
            raise ValueError("W1 Drive run name already exists; refusing to reuse it")
        remote_id = find_or_create_folder(service, parent, args.w1_run_id)
        publish("preparing", drive_run_folder_id=remote_id, output_dir=str(output_dir))
    mirror = DriveMirror(source_dir, remote_id, args.w1_run_id, b1_config["drive_parent_folder_id"], 536870912)
    expected_readbacks = ["control/w1-config.json", "control/w1-queue-config.json",
                          "control/w1-queue-status.json", "preprocessing/feature_validation.json",
                          "preprocessing/feature_inventory.json", "preprocessing/validated_marker_rows.json"]
    if not prepared_status.get("drive_run_folder_id"):
        publish("preparing", drive_run_folder_id=remote_id, output_dir=str(output_dir))
    mirror.sync(w1_control, output_dir)
    if any(key not in mirror.previous for key in expected_readbacks):
        publish("initial_sync_failed", drive_run_folder_id=remote_id,
                error="Drive initial sync did not byte-readback required W1 config/status/validation metadata")
        return 1
    remaining = (deadline - utcnow()).total_seconds()
    runtime_hours = min(float(b1_config.get("max_runtime_hours", 10)),
                        remaining / 3600 - args.final_sync_minutes / 60)
    if runtime_hours * 60 < args.min_remaining_minutes:
        publish("queued", reason="initial Drive readback consumed the minimum W1 window; prepared run remains queued",
                drive_run_folder_id=remote_id, output_dir=str(output_dir))
        mirror.sync(w1_control, output_dir)
        return 0
    w1_config["max_runtime_hours"] = runtime_hours
    write_json(w1_config_path, w1_config)
    contract_write_json(w1_control / "w1-config.json", w1_config)
    publish("launching", drive_run_folder_id=remote_id, output_dir=str(output_dir), device=w1_config["device"])
    mirror.sync(w1_control, output_dir)
    if "control/w1-queue-status.json" not in mirror.previous or "control/w1-config.json" not in mirror.previous:
        publish("initial_sync_failed", drive_run_folder_id=remote_id,
                error="Launch status/config failed initial Drive readback")
        return 1

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if w1_config["device"] == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    log_stream = (w1_control / "w1-training.log").open("a", encoding="utf-8")
    process = subprocess.Popen([sys.executable, str(source_dir / "train_qc200.py"), "--config", str(w1_config_path)],
                               cwd=str(source_dir), env=env, stdout=log_stream, stderr=subprocess.STDOUT)
    write_json(w1_control / "w1-process.json", {"pid": process.pid, "started_utc": iso(utcnow()), "config": str(w1_config_path)})
    publish("running", drive_run_folder_id=remote_id, output_dir=str(output_dir), device=w1_config["device"], pid=process.pid)
    sync_stop = threading.Event()
    sync_error = {"value": None}

    def persistence_loop() -> None:
        while not sync_stop.wait(45):
            try:
                mirror.sync(w1_control, output_dir)
                sync_error["value"] = None
            except Exception as exc:
                sync_error["value"] = f"{type(exc).__name__}: {exc}"
                write_json(w1_control / "persistence-error.json", {"utc": iso(utcnow()), "error": sync_error["value"]})
                Path(w1_config["pause_file"]).write_text("Drive persistence failed\n", encoding="utf-8")

    sync_thread = threading.Thread(target=persistence_loop, name="w1-drive-sync", daemon=True)
    sync_thread.start()
    forced_termination = False
    try:
        while process.poll() is None:
            save_reserve = float(w1_config.get("save_reserve_minutes", 10)) * 60
            if (deadline - utcnow()).total_seconds() <= final_sync_reserve + save_reserve + 60:
                Path(w1_config["pause_file"]).write_text("Entering the absolute deadline checkpoint/sync reserve\n", encoding="utf-8")
            if utcnow() >= deadline:
                Path(w1_config["pause_file"]).write_text("Absolute runtime deadline reached\n", encoding="utf-8")
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    forced_termination = True
                    process.terminate()  # Last resort after giving the trainer a checkpoint opportunity.
                    process.wait()
                break
            try:
                persistence_state = read_json(w1_control / "persistence-state.json")
                synced = datetime.fromisoformat(persistence_state["utc"].replace("Z", "+00:00"))
                if (utcnow() - synced.astimezone(timezone.utc)).total_seconds() > 300:
                    Path(w1_config["pause_file"]).write_text("Persistence has not completed a verified pass for 300 seconds\n", encoding="utf-8")
            except Exception:
                Path(w1_config["pause_file"]).write_text("Persistence state unavailable\n", encoding="utf-8")
            time.sleep(5)
        returncode = process.wait()
    finally:
        sync_stop.set()
        sync_thread.join()
        log_stream.close()
    train_status = read_json(output_dir / "run-status.json") if (output_dir / "run-status.json").is_file() else {}
    if returncode == 0 and train_status.get("state") == "complete_train_dev":
        final_state = "complete_train_dev"
    elif returncode == 0 and train_status.get("state") == "paused_runtime":
        final_state = "paused_runtime"
    else:
        final_state = "failed"
    publish(final_state, drive_run_folder_id=remote_id, output_dir=str(output_dir), returncode=returncode,
            training_state=train_status.get("state"), persistence_error=sync_error["value"],
            forced_termination=forced_termination, eval_features_or_scores_accessed=False)
    try:
        mirror.sync(w1_control, output_dir)
    except Exception as exc:
        write_json(w1_control / "persistence-error.json", {"utc": iso(utcnow()), "error": f"final sync: {type(exc).__name__}: {exc}"})
        publish(final_state, drive_run_folder_id=remote_id, output_dir=str(output_dir), returncode=returncode,
                training_state=train_status.get("state"), persistence_error=f"final sync: {type(exc).__name__}: {exc}")
    return 0 if final_state in ("complete_train_dev", "paused_runtime") and not sync_error["value"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
