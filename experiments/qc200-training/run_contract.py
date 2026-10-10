"""Explicit, hash-free identities for the QC200 experiments."""
from __future__ import annotations
import json
import re
from pathlib import Path

SOURCE_VERSION = "qc200-20261010-v1"

def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = ("run_id", "feature_version", "preproc_dir", "marker_dir", "output_dir")
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise ValueError(f"Missing configuration fields: {missing}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*",config["run_id"]):
        raise ValueError("run_id must be a safe single directory name")
    if config.get("sample_rate_hz", 200) != 200:
        raise ValueError("This experiment supports 200 Hz only")
    if config.get("variant", "B1") not in ("B1", "W1"):
        raise ValueError("variant must be B1 or W1")
    return config

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)

def rows_contract(rows):
    """Store exact private identities rather than reducing them to a hash."""
    return [[rel, int(label), patient] for rel, label, patient in rows]

def sampler_statistics(rows):
    from collections import Counter
    counts = Counter(patient for _, _, patient in rows)
    total_mass = sum(1 / counts[patient] for _, _, patient in rows)
    positive_mass = sum(label / counts[patient] for _, label, patient in rows)
    q = positive_mass / total_mass
    if not 0 < q < 1:
        raise ValueError("Sampler must contain both classes")
    return {"patients": len(counts), "clips": len(rows), "positive": sum(r[1] for r in rows),
            "negative": sum(1-r[1] for r in rows), "positive_probability": q,
            "candidate_pos_weight": (1-q)/q,
            "policy": "patient-total-equal replacement on seeded balanced pool"}

def verify_feature_inventory(config, validation_root):
    """Recheck cache location, sizes, modification times and producer metadata."""
    import h5py
    root=Path(config["preproc_dir"]).resolve()
    validation_root=Path(validation_root)
    report=json.loads((validation_root/"feature_validation.json").read_text())
    if report.get("feature_root")!=str(root):
        raise ValueError("Feature cache location differs from validated cache")
    for item in json.loads((validation_root/"feature_inventory.json").read_text()):
        path=root/item["row"]
        if not path.is_file(): raise ValueError("Validated feature missing")
        stat=path.stat()
        if stat.st_size!=item["bytes"] or stat.st_mtime_ns!=item["mtime_ns"]:
            raise ValueError("Feature cache metadata changed after validation")
        with h5py.File(path,"r") as record:
            if str(record.attrs.get("feature_version",""))!=item["feature_version"] or str(record.attrs.get("producer_version",""))!=item["producer_version"]:
                raise ValueError("Feature producer metadata changed after validation")
