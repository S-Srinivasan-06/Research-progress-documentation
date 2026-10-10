"""Stage 200 Hz feature and marker shards from Drive into Kaggle temporary storage."""
from __future__ import annotations
import argparse, json, os, shutil, tarfile, time
from pathlib import Path, PurePosixPath

RETRIES = 3
RESERVE_BYTES = 2 * 1024**3
MARKER_NAMES = {f"{s}Set_seq2seq_12s_{k}.txt" for s in ("train", "dev", "test") for k in ("sz", "nosz")}


def safe_name(name: str) -> PurePosixPath:
    p = PurePosixPath(name)
    if p.is_absolute() or not p.parts or ".." in p.parts or "\\" in name:
        raise ValueError(f"Unsafe TAR path: {name!r}")
    return p


def download(entry: dict, target: Path) -> Path:
    import gdown
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == entry["size"]:
        return target
    part = target.with_suffix(target.suffix + ".part")
    last = None
    for attempt in range(RETRIES):
        try:
            gdown.download(id=entry["id"], output=str(part), quiet=True, resume=True)
            actual = part.stat().st_size if part.is_file() else 0
            if actual != entry["size"]:
                part.unlink(missing_ok=True)
                raise IOError(f"size mismatch: expected {entry['size']}, got {actual}")
            part.replace(target)
            return target
        except Exception as e:
            last = e
            msg = str(e).lower()
            if "too many users" in msg or "quota" in msg or "download quota" in msg:
                break
            if attempt + 1 < RETRIES:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Drive download failed for {entry['name']}; resumable partial at {part}: {last}") from last


def resolve_input_archive(entry: dict, input_dir: Path) -> Path:
    """Find exactly one immutable mounted archive with the expected name and size."""
    root = input_dir.resolve(strict=True)
    matches = [p for p in root.rglob(entry["name"]) if p.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one mounted {entry['name']} under {root}; found {len(matches)}")
    archive = matches[0].resolve(strict=True)
    if not archive.is_relative_to(root):
        raise ValueError(f"Mounted archive resolves outside input directory: {entry['name']}")
    actual = archive.stat().st_size
    if actual != entry["size"]:
        raise IOError(f"Mounted archive size mismatch for {entry['name']}: expected {entry['size']}, got {actual}")
    return archive


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def stage_archive(item: dict, archive: Path, feature_root: Path, marker_shards: Path,
                  archive_meta: Path, batch: int, owners: dict[str, int]) -> dict:
    """Stream one TAR, preserving H5 paths and isolating archive-local metadata."""
    written, files, marker_paths = 0, [], []
    seen: set[str] = set()
    feature = item["role"] == "features"
    with tarfile.open(archive, mode="r:gz") as tar:
        for member in tar:
            rel = safe_name(member.name)
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"Non-regular TAR member rejected: {member.name!r}")
            if rel.suffix.lower() in {".tar", ".gz", ".tgz", ".zip", ".7z"}:
                raise ValueError(f"Nested archive rejected: {member.name!r}")
            rel_s = rel.as_posix()
            if rel_s in seen:
                raise ValueError(f"Duplicate TAR member: {member.name!r}")
            seen.add(rel_s)
            is_marker = rel.name in MARKER_NAMES
            if feature and rel.suffix.lower() != ".h5":
                # Preserve per-shard reports/embedded markers outside the training tree.
                dest = archive_meta / f"batch-{batch:05d}" / item["role"] / rel_s
            elif feature:
                owner = owners.get(rel_s)
                if owner is not None and owner != batch:
                    raise ValueError(f"Feature path collision across batches {owner} and {batch}: {rel_s}")
                dest = feature_root.joinpath(*rel.parts)
            elif is_marker:
                dest = marker_shards / f"batch-{batch:05d}" / rel.name
                marker_paths.append(rel.name)
            else:
                dest = archive_meta / f"batch-{batch:05d}" / item["role"] / rel_s
            if shutil.disk_usage(feature_root).free < member.size + RESERVE_BYTES:
                raise OSError(f"Insufficient /kaggle/temp space for {member.name!r}; retain {RESERVE_BYTES} byte reserve")
            dest.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(member)
            if src is None:
                raise IOError(f"Cannot read TAR member: {member.name!r}")
            tmp = dest.with_name(dest.name + ".part")
            with src, tmp.open("wb") as sink:
                shutil.copyfileobj(src, sink, length=1024 * 1024)
            if tmp.stat().st_size != member.size:
                tmp.unlink(missing_ok=True)
                raise IOError(f"Extracted size mismatch: {member.name!r}")
            # Existing same-batch path is only from an interrupted retry; replace it atomically.
            if feature and rel.suffix.lower() == ".h5" and dest.exists() and owners.get(rel_s) != batch:
                raise ValueError(f"Unexpected existing feature path: {rel_s}")
            tmp.replace(dest)
            files.append({"path": str(dest), "relative": rel_s, "bytes": member.size})
            written += member.size
    if not feature and set(marker_paths) != MARKER_NAMES:
        raise ValueError(f"Marker TAR {item['name']} did not contain exactly the six expected lists")
    return {"id": item["id"], "name": item["name"], "role": item["role"], "extracted_bytes": written,
            "file_count": len(files), "files": files, "marker_names": marker_paths}


def verify_record(record: dict) -> bool:
    files = record.get("files", [])
    return (len(files) == record.get("file_count") and
            sum(x["bytes"] for x in files) == record.get("extracted_bytes") and
            all(Path(x["path"]).is_file() and Path(x["path"]).stat().st_size == x["bytes"] for x in files))


def feature_inventory(archive: Path) -> list[str]:
    paths = []
    with tarfile.open(archive, mode="r:gz") as tar:
        for member in tar:
            rel = safe_name(member.name)
            if member.isdir():
                continue
            if not member.isfile() or rel.suffix.lower() in {".tar", ".gz", ".tgz", ".zip", ".7z"}:
                raise ValueError(f"Unsafe or nested TAR member: {member.name!r}")
            if rel.suffix.lower() == ".h5":
                paths.append(rel.as_posix())
    if len(paths) != len(set(paths)):
        raise ValueError("Feature TAR contains duplicate paths")
    return paths


def merge_markers(marker_shards: Path, marker_root: Path, expected_rows: int) -> dict:
    rows: dict[str, tuple[int, str, str]] = {}
    per_file: dict[str, list[str]] = {name: [] for name in sorted(MARKER_NAMES)}
    patients: dict[str, str] = {}
    for name in sorted(MARKER_NAMES):
        for src in sorted(marker_shards.glob(f"batch-*/{name}")):
            with src.open(encoding="utf-8") as stream:
                for n, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        marker, label_text = line.rsplit(",", 1)
                        label = int(label_text.strip())
                    except ValueError as e:
                        raise ValueError(f"Malformed marker at {src}:{n}") from e
                    marker = marker.strip().replace("\\", "/")
                    parts = PurePosixPath(marker).parts
                    want_split = name.split("Set_", 1)[0]
                    expected_prefix = "eval" if want_split == "test" else want_split
                    expected_label = 1 if "_sz.txt" in name else 0
                    if label != expected_label or len(parts) < 2 or parts[0] != expected_prefix:
                        raise ValueError(f"Unexpected split/label in {src}:{n}: {marker}, {label}")
                    if marker in rows:
                        raise ValueError(f"Duplicate clip marker across shards: {marker}")
                    patient = parts[1]
                    old_split = patients.setdefault(patient, want_split)
                    if old_split != want_split:
                        raise ValueError(f"Patient appears in multiple splits: {patient}: {old_split}, {want_split}")
                    rows[marker] = (label, name, str(src))
                    per_file[name].append(f"{marker}, {label}\n")
    if len(rows) != expected_rows:
        raise ValueError(f"Marker coverage mismatch: expected {expected_rows}, found {len(rows)}")
    marker_root.mkdir(parents=True, exist_ok=True)
    for name, lines in per_file.items():
        tmp = marker_root / f"{name}.part"
        with tmp.open("w", encoding="utf-8", newline="") as out:
            out.writelines(lines)
        tmp.replace(marker_root / name)
    return {"marker_rows": len(rows), "patients": len(patients), "files": {k: len(v) for k, v in per_file.items()}}


def run(manifest_path: Path, root: Path, batch: int | None = None, probe: bool = False,
        status_dir: Path | None = None, input_dir: Path | None = None) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest["files"]
    by_batch: dict[int, list[dict]] = {}
    for item in files:
        by_batch.setdefault(int(item["batch"]), []).append(item)
    if batch is not None and batch not in by_batch:
        raise ValueError(f"Unknown batch {batch}")
    selected = [batch] if batch is not None else sorted(by_batch)
    status_dir = status_dir or Path(os.environ.get("TUSZ200_STATUS_DIR", "/kaggle/working/tusz200-staging-status"))
    status_dir.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    stage_root = root / "stage"
    downloads = stage_root / "downloads"
    features = root / "features"
    marker_shards = stage_root / "marker-shards"
    archive_meta = stage_root / "archive-meta"
    checkpoints = stage_root / "checkpoints"
    for p in (downloads, features, marker_shards, archive_meta, checkpoints):
        p.mkdir(parents=True, exist_ok=True)
    if probe:
        entry = min((x for x in by_batch[selected[0]] if x["role"] == "markers"), key=lambda x: x["size"])
        started = time.monotonic()
        source = "private Kaggle Dataset" if input_dir else "Drive/gdown"
        print(f"Archive probe: reading {entry['name']} ({entry['size']} bytes) from {source}", flush=True)
        got = resolve_input_archive(entry, input_dir) if input_dir else download(entry, downloads / entry["name"])
        result = {"probe": "ok", "source": source, "name": entry["name"], "bytes": got.stat().st_size,
                  "elapsed_seconds": round(time.monotonic() - started, 1)}
        print(f"Drive probe complete: {result['bytes']} bytes in {result['elapsed_seconds']}s", flush=True)
        return result
    # Rebuild path ownership from durable shard checkpoints; no archive is reread for completed shards.
    owners: dict[str, int] = {}
    for cp in sorted(checkpoints.glob("batch-*.json")):
        state = json.loads(cp.read_text(encoding="utf-8"))
        if state.get("complete") and all(verify_record(x) for x in state.get("files", [])):
            for rec in state["files"]:
                if rec["role"] == "features":
                    for f in rec["files"]:
                        owners[f["relative"]] = state["batch"]
        elif not state.get("complete"):
            for rel in state.get("inflight", {}).values():
                for path in rel:
                    owners[path] = state["batch"]
    reports = []
    for batch_index, b in enumerate(selected, 1):
        pair = by_batch[b]
        if {x["role"] for x in pair} != {"features", "markers"}:
            raise ValueError(f"Manifest missing feature/marker pair for batch {b}")
        cp = checkpoints / f"batch-{b:05d}.json"
        prior = json.loads(cp.read_text(encoding="utf-8")) if cp.is_file() else {"batch": b, "files": []}
        records = {x["id"]: x for x in prior.get("files", []) if verify_record(x)}
        if prior.get("complete") and len(records) == 2:
            print(f"[{batch_index}/{len(selected)}] batch {b:05d} already staged; verified checkpoint", flush=True)
            reports.append(prior)
            continue
        for item in sorted(pair, key=lambda x: (x["role"] != "markers", x["role"])):
            if item["id"] in records:
                continue
            if input_dir is None:
                need = item["size"] + RESERVE_BYTES
                if shutil.disk_usage(root).free < need:
                    raise OSError(f"Insufficient /kaggle/temp space to download {item['name']}: need {need}")
            started = time.monotonic()
            status_path = status_dir / f"batch-{b:05d}-{item['role']}.json"
            save_json(status_path, {"batch": b, "batch_index": batch_index, "batch_total": len(selected),
                                   "role": item["role"], "name": item["name"], "status": "downloading",
                                   "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            source = "private Kaggle Dataset" if input_dir else "Drive/gdown"
            action = "reading" if input_dir else "downloading"
            print(f"[{batch_index}/{len(selected)}] batch {b:05d}: {action} {item['name']} ({item['size']} bytes) from {source}", flush=True)
            archive = resolve_input_archive(item, input_dir) if input_dir else download(item, downloads / item["name"])
            print(f"[{batch_index}/{len(selected)}] batch {b:05d}: verified {archive.stat().st_size} bytes; extracting {item['name']}", flush=True)
            if item["role"] == "features":
                inventory = feature_inventory(archive)
                collisions = [(p, owners[p]) for p in inventory if p in owners and owners[p] != b]
                if collisions:
                    raise ValueError(f"Feature path collision across batches: {collisions[0]}")
                for p in inventory:
                    owners[p] = b
                inflight = dict(prior.get("inflight", {}))
                inflight[item["id"]] = inventory
                save_json(cp, {"batch": b, "complete": False, "files": list(records.values()), "inflight": inflight})
            save_json(status_path, {"batch": b, "batch_index": batch_index, "batch_total": len(selected),
                                    "role": item["role"], "name": item["name"], "status": "extracting",
                                    "source": source, "source_bytes": archive.stat().st_size,
                                    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            record = stage_archive(item, archive, features, marker_shards, archive_meta, b, owners)
            # Persist a verified per-shard checkpoint before deleting the downloaded temporary archive.
            if not verify_record(record):
                raise IOError(f"Staged files failed verification for {item['name']}")
            records[item["id"]] = record
            inflight = dict(prior.get("inflight", {}))
            inflight.pop(item["id"], None)
            prior["inflight"] = inflight
            if item["role"] == "features":
                for f in record["files"]:
                    owners[f["relative"]] = b
            save_json(cp, {"batch": b, "complete": False, "files": list(records.values()), "inflight": inflight})
            if input_dir is None:
                archive.unlink()
            elapsed = round(time.monotonic() - started, 1)
            print(f"[{batch_index}/{len(selected)}] batch {b:05d}: staged {item['name']} ({record['extracted_bytes']} extracted bytes, {record['file_count']} files) in {elapsed}s", flush=True)
            save_json(status_path, {
                "batch": b, "batch_index": batch_index, "batch_total": len(selected), "role": item["role"],
                "name": item["name"], "status": "complete", "source": source, "source_bytes": item["size"],
                "extracted_bytes": record["extracted_bytes"], "file_count": record["file_count"],
                "elapsed_seconds": elapsed, "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        state = {"batch": b, "complete": True, "files": list(records.values())}
        save_json(cp, state)
        reports.append(state)
    if batch is None:
        marker_report = merge_markers(marker_shards, root / "markers", expected_rows=373792)
        result = {"batches_complete": len(reports), "feature_root": str(features), "marker_root": str(root / "markers"),
                  "marker_report": marker_report, "feature_files": sum(x["file_count"] for r in reports for x in r["files"] if x["role"] == "features"),
                  "feature_bytes": sum(x["extracted_bytes"] for r in reports for x in r["files"] if x["role"] == "features")}
        save_json(root / "stage-report.json", result)
        return result
    return {"batch": batch, "complete": True, "checkpoint": str(checkpoints / f"batch-{batch:05d}.json")}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path(__file__).with_name("manifest.json"))
    ap.add_argument("--batch", type=int)
    ap.add_argument("--all", action="store_true", help="stage all 40 pairs and consolidate marker lists")
    ap.add_argument("--root", type=Path, default=Path("/kaggle/temp/tusz200"))
    ap.add_argument("--input-dir", type=Path, default=None, help="read-only private Kaggle Dataset mount; skips gdown when supplied")
    ap.add_argument("--status-dir", type=Path, default=None, help="small live progress JSONs; default /kaggle/working/tusz200-staging-status")
    ap.add_argument("--probe", action="store_true", help="download and size-check only the smallest marker TAR")
    a = ap.parse_args()
    if not a.all and a.batch is None and not a.probe:
        ap.error("specify --all, --batch N, or --probe")
    print(json.dumps(run(a.manifest, a.root, None if a.all else a.batch, a.probe, a.status_dir, a.input_dir), indent=2))
