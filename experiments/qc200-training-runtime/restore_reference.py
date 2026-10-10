"""Restore the pinned public DGDCN reference plus local runtime repairs.

This helper downloads only the files needed by the frozen QC200 trainer. The
upstream repository has no license declaration in the inspected snapshot; this
runtime restore does not grant permission to redistribute its contents.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen


UPSTREAM = "https://github.com/Open-EXG/DGDCN-EEG-Seizure-Detection"
RAW_BASE = "https://raw.githubusercontent.com/Open-EXG/DGDCN-EEG-Seizure-Detection"
COMMIT = "96c7bee49c240124735ffc2ab2ad118d27a1ff97"
MODEL_LF_LINES = (
    4, 276, 293, 294, 295, 296, 322, 337, 338, 339,
    427, 444, 445, 446, 447, 473, 488, 489, 490,
    579, 596, 597, 598, 599, 625, 640, 641, 642,
)
FILES = {
    "constants.py": {
        "upstream_path": "constants.py", "upstream_bytes": 516,
        "active_bytes": 544, "line_endings": "crlf",
    },
    "data/electrode_graph/adj_mx_3d.pkl": {
        "upstream_path": "data/electrode_graph/adj_mx_3d.pkl",
        "upstream_bytes": 2105, "active_bytes": 2105, "line_endings": "binary",
    },
    "model/DGDCN/lib/__init__.py": {
        "upstream_path": "model/DGDCN/lib/__init__.py",
        "upstream_bytes": 0, "active_bytes": 0, "line_endings": "binary",
    },
    "model/DGDCN/lib/metrics.py": {
        "upstream_path": "model/DGDCN/lib/metrics.py",
        "upstream_bytes": 541, "active_bytes": 557, "line_endings": "crlf",
    },
    "model/DGDCN/lib/utils.py": {
        "upstream_path": "model/DGDCN/lib/utils.py",
        "upstream_bytes": 18859, "active_bytes": 19364, "line_endings": "crlf",
    },
    "model/DGDCN/model/DGDCN_r.py": {
        "upstream_path": "model/DGDCN/model/DGDCN_r.py",
        "upstream_bytes": 26689, "active_bytes": 27578,
        "line_endings": "mixed", "line_count": 656,
        "lf_line_indices": MODEL_LF_LINES,
    },
}
MAX_DOWNLOAD_BYTES = 1_000_000


def download(path: str, expected_size: int) -> bytes:
    url = f"{RAW_BASE}/{COMMIT}/{path}"
    with urlopen(url, timeout=60) as response:
        content = response.read(MAX_DOWNLOAD_BYTES + 1)
    if len(content) > MAX_DOWNLOAD_BYTES or len(content) != expected_size:
        raise ValueError(f"Pinned upstream size mismatch for {path}")
    return content


def restore_active_line_endings(path: str, content: bytes) -> bytes:
    spec = FILES[path]
    policy = spec["line_endings"]
    if policy == "binary":
        result = content
    elif policy == "crlf":
        if b"\r" in content:
            raise ValueError(f"Unexpected carriage return in pinned source: {path}")
        result = content.replace(b"\n", b"\r\n")
    elif policy == "mixed":
        if b"\r" in content:
            raise ValueError(f"Unexpected carriage return in patched source: {path}")
        lines = content.splitlines(keepends=True)
        if len(lines) != spec["line_count"]:
            raise ValueError(f"Patched source line count differs for {path}")
        lf_indices = set(spec["lf_line_indices"])
        if any(index >= len(lines) or not lines[index].endswith(b"\n") for index in lf_indices):
            raise ValueError(f"Mixed-newline line index is invalid for {path}")
        result = b"".join(
            line if index in lf_indices else line[:-1] + b"\r\n"
            for index, line in enumerate(lines)
        )
    else:
        raise ValueError(f"Unsupported line-ending policy for {path}: {policy}")
    if len(result) != spec["active_bytes"]:
        raise ValueError(f"Reconstructed active byte size differs for {path}")
    return result


def replace_exact(text: str, old: str, new: str, expected: int) -> str:
    count = text.count(old)
    if count != expected:
        raise ValueError(f"Expected {expected} source occurrence(s), found {count}: {old!r}")
    return text.replace(old, new)


def repair_model(content: bytes) -> bytes:
    text = content.decode("utf-8").replace("\r\n", "\n")
    text = replace_exact(
        text,
        "from model.DGDCN.lib.utils_DGDCN import scaled_Laplacian, cheb_polynomial",
        "from model.DGDCN.lib.utils import scaled_Laplacian, cheb_polynomial",
        1,
    )
    text = replace_exact(
        text,
        "self.layer_0 = nn.Linear(19*12, 128)",
        "self.layer_0 = nn.Linear(num_of_vertices * num_for_predict, 128)",
        3,
    )
    text = replace_exact(text, "        print(x.shape)\n", "", 3)
    text = replace_exact(
        text,
        "num_of_vertices,DEVICE=torch.device('cuda:1')):",
        "num_of_vertices, DEVICE=None):",
        3,
    )
    text = replace_exact(
        text,
        "    :return:\n    '''\n    L_tilde = scaled_Laplacian(adj_mx)\n",
        "    :return:\n    '''\n"
        "    if DEVICE is None:\n"
        "        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
        "    L_tilde = scaled_Laplacian(adj_mx)\n",
        3,
    )
    return text.encode("utf-8")


def restore(destination: Path) -> dict:
    root = destination.resolve()
    root.mkdir(parents=True, exist_ok=True)
    payloads = {}
    for relative, spec in FILES.items():
        content = download(spec["upstream_path"], spec["upstream_bytes"])
        if relative.endswith("DGDCN_r.py"):
            content = repair_model(content)
        content = restore_active_line_endings(relative, content)
        payloads[relative] = content

    for relative, content in payloads.items():
        target = root / relative
        if target.is_symlink():
            raise ValueError(f"Refusing symlink target: {target}")
        if target.exists() and target.read_bytes() != content:
            raise FileExistsError(f"Existing reference file differs: {target}")

    restored = []
    for relative, content in payloads.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            fd, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".part", dir=target.parent)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(content)
                Path(temporary).replace(target)
            finally:
                Path(temporary).unlink(missing_ok=True)
        restored.append({"path": relative, "bytes": len(content)})

    receipt = {
        "upstream_repository": UPSTREAM,
        "upstream_commit": COMMIT,
        "patch": "qc200-runtime-repairs-v2-exact-newlines",
        "files": restored,
        "license_note": "No upstream license identified; this receipt grants no redistribution rights.",
    }
    receipt_path = root / "REFERENCE_RESTORE.json"
    encoded = (json.dumps(receipt, indent=2) + "\n").encode("utf-8")
    if receipt_path.exists() and receipt_path.read_bytes() != encoded:
        raise FileExistsError(f"Existing restore receipt differs: {receipt_path}")
    if not receipt_path.exists():
        receipt_path.write_bytes(encoded)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(restore(args.destination), indent=2))


if __name__ == "__main__":
    main()
