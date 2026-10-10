"""Read TUSZ v2.0.6 annotations and prepare fresh DGDCN clip markers.

This module deliberately reads EDF headers only. It does not decode or write
EEG samples. The output marker files follow the archived DGDCN loader contract.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Iterable


SPLITS = ("train", "dev", "eval")
MARKER_SPLIT = {"train": "train", "dev": "dev", "eval": "test"}
CHANNELS = (
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ",
)
SEIZURE_LABELS = {"seiz", "fnsz", "gnsz", "spsz", "cpsz", "absz", "tnsz", "tcsz", "mysz"}


def union_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted(intervals)
    merged: list[list[float]] = []
    for start, stop in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    return [(start, stop) for start, stop in merged]


def _header_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Return normalized column names and rows from a comment-prefixed CSV."""
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        lines = stream.readlines()
    header_index = next(
        (i for i, line in enumerate(lines) if line.strip().lower().startswith("channel,start_time,stop_time,label")),
        None,
    )
    if header_index is None:
        raise ValueError(f"No TUSZ annotation CSV header found: {path}")
    reader = csv.DictReader(lines[header_index:])
    rows = list(reader)
    return [str(name).strip().lower() for name in (reader.fieldnames or [])], rows


def seizure_intervals(annotation_path: str | Path) -> list[tuple[float, float]]:
    """Read seizure intervals and union overlaps across per-channel rows."""
    path = Path(annotation_path)
    headers, rows = _header_rows(path)
    required = {"start_time", "stop_time", "label"}
    if not required.issubset(headers):
        raise ValueError(f"Missing required annotation columns in {path}: {sorted(required - set(headers))}")
    intervals: list[tuple[float, float]] = []
    for row in rows:
        label = (row.get("label") or "").strip().lower()
        if label not in SEIZURE_LABELS:
            continue
        start, stop = float(row["start_time"]), float(row["stop_time"])
        if start >= 0 and stop == start:
            warnings.warn(f"Ignored zero-duration seizure annotation {start}, {stop} in {path}", RuntimeWarning)
            continue
        if start < 0 or stop < start:
            raise ValueError(f"Invalid seizure interval {start}, {stop} in {path}")
        intervals.append((start, stop))
    return union_intervals(intervals)


def clip_label(start: float, stop: float, intervals: Iterable[tuple[float, float]], threshold_seconds: float = 1.0) -> int:
    """Positive iff the union of annotated seizure time overlaps by >= threshold."""
    overlap = sum(max(0.0, min(stop, event_stop) - max(start, event_start))
                  for event_start, event_stop in union_intervals(intervals))
    return int(overlap + 1e-9 >= threshold_seconds)


def edf_header(path: str | Path) -> tuple[list[str], float]:
    """Read EDF fixed/signal headers for labels and duration; never reads samples."""
    path = Path(path)
    with path.open("rb") as stream:
        fixed = stream.read(256)
        if len(fixed) != 256:
            raise ValueError(f"Truncated EDF fixed header: {path}")
        try:
            header_bytes = int(fixed[184:192].decode("ascii").strip())
            record_count = int(fixed[236:244].decode("ascii").strip())
            record_duration = float(fixed[244:252].decode("ascii").strip())
            signal_count = int(fixed[252:256].decode("ascii").strip())
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError(f"Invalid EDF fixed header: {path}") from exc
        if signal_count <= 0 or header_bytes < 256 + signal_count * 16:
            raise ValueError(f"Invalid EDF signal/header count: {path}")
        labels = [stream.read(16).decode("ascii", "replace").strip() for _ in range(signal_count)]
    duration = record_count * record_duration if record_count >= 0 else 0.0
    return labels, duration


def channel_reference(labels: Iterable[str]) -> tuple[str, list[str]]:
    """Check 19 requested electrodes and report their common EDF reference suffix."""
    found: dict[str, list[str]] = defaultdict(list)
    for raw in labels:
        clean = re.sub(r"\s+", " ", raw.strip()).upper()
        if not clean.startswith("EEG ") or "-" not in clean:
            continue
        electrode, reference = clean[4:].rsplit("-", 1)
        if electrode in CHANNELS:
            found[electrode].append(reference)
    missing = [channel for channel in CHANNELS if len(found[channel]) != 1]
    if missing:
        raise ValueError("Missing or duplicate required channels: " + ", ".join(missing))
    refs = {found[channel][0] for channel in CHANNELS}
    if len(refs) != 1:
        raise ValueError(f"Mixed reference suffixes across 19 channels: {sorted(refs)}")
    return next(iter(refs)), [f"EEG {channel}-{next(iter(refs))}" for channel in CHANNELS]


def _annotation_duration(path: Path) -> float | None:
    with path.open("r", encoding="utf-8-sig") as stream:
        for line in stream:
            match = re.match(r"\s*#\s*duration\s*=\s*([0-9.]+)\s+secs", line, re.I)
            if match:
                return float(match.group(1))
    return None


def prepare(dataset_root: Path, output_dir: Path, clip_lengths: tuple[int, ...],
            min_overlap: float, limit_per_split: int | None, dry_run: bool) -> dict:
    edf_root = dataset_root / "edf"
    if not edf_root.is_dir():
        raise FileNotFoundError(f"Expected TUSZ release root containing edf/: {dataset_root}")
    files_by_split = {
        split: sorted((edf_root / split).rglob("*.edf"))
        for split in SPLITS
    }
    if any(not files for files in files_by_split.values()):
        raise ValueError("Expected non-empty official edf/train, edf/dev, and edf/eval folders")
    if limit_per_split is not None:
        if limit_per_split < 1:
            raise ValueError("limit_per_split must be positive")
        files_by_split = {split: files[:limit_per_split]
                          for split, files in files_by_split.items()}

    patient_sets: dict[str, set[str]] = {split: set() for split in SPLITS}
    marker_lines: dict[tuple[str, int, int], list[str]] = defaultdict(list)
    manifest: list[dict] = []
    class_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for split in SPLITS:
        for edf_path in files_by_split[split]:
            relative = edf_path.relative_to(edf_root)
            if len(relative.parts) < 5 or relative.parts[0] != split:
                raise ValueError(f"Unexpected TUSZ EDF path layout: {relative}")
            patient_id = relative.parts[1]
            patient_sets[split].add(patient_id)
            annotation_path = edf_path.with_name(edf_path.stem + ".csv_bi")
            if not annotation_path.is_file():
                raise FileNotFoundError(f"Missing current binary annotation: {annotation_path}")
            intervals = seizure_intervals(annotation_path)
            labels, header_duration = edf_header(edf_path)
            reference, _ = channel_reference(labels)
            duration = _annotation_duration(annotation_path) or header_duration
            if duration <= 0 or header_duration > 0 and abs(duration - header_duration) > 1.0:
                raise ValueError(f"EDF/annotation duration mismatch for {relative}: {header_duration} vs {duration}")
            row = {
                "split": split, "patient_id": patient_id,
                "edf": relative.as_posix(), "annotation": annotation_path.relative_to(edf_root).as_posix(),
                "duration_seconds": duration, "annotation_intervals": len(intervals),
                "reference_suffix": reference, "channels": list(CHANNELS),
            }
            manifest.append(row)
            for clip_length in clip_lengths:
                marker_split = MARKER_SPLIT[split]
                clip_count = int(duration // clip_length)
                for clip_idx in range(clip_count):
                    start = clip_idx * clip_length
                    label = clip_label(start, start + clip_length, intervals, min_overlap)
                    category = "sz" if label else "nosz"
                    marker_name = f"{relative.as_posix()}_{clip_idx}.h5, {label}\n"
                    marker_lines[(marker_split, clip_length, category)].append(marker_name)
                    class_counts[f"{marker_split}/{clip_length}s"][category] += 1

    overlaps = {
        f"{left}-{right}": sorted(patient_sets[left] & patient_sets[right])
        for i, left in enumerate(SPLITS) for right in SPLITS[i + 1:]
    }
    overlap_found = {key: value for key, value in overlaps.items() if value}
    if overlap_found:
        raise ValueError(f"Patient identifiers overlap between official split folders: {overlap_found}")

    report = {
        "release": "TUSZ v2.0.6",
        "annotation_source": "sibling .csv_bi; label=seiz",
        "label_rule": f"unioned seizure overlap >= {min_overlap:g} second in a complete non-overlapping clip",
        "clip_lengths_seconds": list(clip_lengths),
        "signal_preprocessing": "not performed; EDF headers only",
        "reference_policy": "electrode names are canonicalized; original LE/REF suffix is retained per recording; no rereferencing is performed",
        "official_split_patient_counts_scanned": {split: len(ids) for split, ids in patient_sets.items()},
        "patient_overlap": overlaps,
        "recordings_scanned": len(manifest),
        "clips_by_split_and_length": class_counts,
        "limited_scan": limit_per_split is not None,
    }
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "recordings.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for row in manifest:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        (output_dir / "metadata.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        for clip_length in clip_lengths:
            for split in ("train", "dev", "test"):
                for category in ("sz", "nosz"):
                    target = output_dir / f"{split}Set_seq2seq_{clip_length}s_{category}.txt"
                    target.write_text("".join(marker_lines[(split, clip_length, category)]), encoding="utf-8")
    return report


def compute_train_stats(marker_dir: Path, preproc_dir: Path, clip_length: int,
                        population: str = "balanced", sampling_ratio: float = 1.0,
                        seed: int = 123) -> dict:
    """Stream train-only FFT clips into DGDCN's (1, nodes, 1) scaler.

    ``balanced`` mirrors the loader's seeded positive/negative undersampling;
    ``all`` uses every generated train marker.
    """
    import h5py
    import numpy as np

    names = (
        f"trainSet_seq2seq_{clip_length}s_sz.txt",
        f"trainSet_seq2seq_{clip_length}s_nosz.txt",
    )
    seizure_rows: list[str] = []
    background_rows: list[str] = []
    for name in names:
        marker_file = marker_dir / name
        if not marker_file.is_file():
            raise FileNotFoundError(f"Missing freshly generated training marker file: {marker_file}")
        target = seizure_rows if name.endswith("_sz.txt") else background_rows
        target.extend(marker_file.read_text(encoding="utf-8").splitlines())
    if population == "balanced":
        if sampling_ratio <= 0:
            raise ValueError("stats_sampling_ratio must be positive")
        rng = np.random.RandomState(seed)
        num_data_points = int(sampling_ratio * len(seizure_rows))
        seizure_indices = np.arange(len(seizure_rows))
        rng.shuffle(seizure_indices)
        seizure_rows = [seizure_rows[i] for i in seizure_indices[:num_data_points]]
        rng.shuffle(background_rows)
        background_rows = background_rows[:num_data_points]
    elif population != "all":
        raise ValueError("stats population must be 'balanced' or 'all'")
    paths = []
    for line in seizure_rows + background_rows:
        relative_h5, _label = line.rsplit(",", 1)
        paths.append(preproc_dir / relative_h5.strip())
    if not paths:
        raise ValueError("No training clips found; cannot compute normalization statistics")

    total = None
    total_squares = None
    count = 0
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing preprocessed training clip: {path}")
        with h5py.File(path, "r") as h5_file:
            data = np.asarray(h5_file["clip"], dtype=np.float64)
        if data.ndim != 3 or data.shape[1] != len(CHANNELS):
            raise ValueError(f"Expected clip shape (time, {len(CHANNELS)}, features), got {data.shape} in {path}")
        by_channel = data.transpose(1, 0, 2).reshape(len(CHANNELS), -1)
        batch_sum = by_channel.sum(axis=1)
        batch_squares = np.square(by_channel).sum(axis=1)
        if total is None:
            total = np.zeros(len(CHANNELS), dtype=np.float64)
            total_squares = np.zeros(len(CHANNELS), dtype=np.float64)
        total += batch_sum
        total_squares += batch_squares
        count += by_channel.shape[1]

    mean = total / count
    variance = np.maximum(total_squares / count - np.square(mean), 1e-12)
    std = np.sqrt(variance)
    means_path = marker_dir / f"means_seq2seq_fft_{clip_length}s_szdetect_single.pkl"
    stds_path = marker_dir / f"stds_seq2seq_fft_{clip_length}s_szdetect_single.pkl"
    marker_dir.mkdir(parents=True, exist_ok=True)
    with means_path.open("wb") as stream:
        pickle.dump(mean.reshape(1, len(CHANNELS), 1), stream)
    with stds_path.open("wb") as stream:
        pickle.dump(std.reshape(1, len(CHANNELS), 1), stream)
    return {
        "mean_file": str(means_path), "std_file": str(stds_path),
        "population": population, "seed": seed,
        "sampling_ratio": sampling_ratio if population == "balanced" else None,
        "positive_clips_used": len(seizure_rows), "background_clips_used": len(background_rows),
        "feature_values_per_channel": count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="TUSZ v2.0.6 release directory containing edf/train, edf/dev, edf/eval")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Fresh marker directory; do not point at archived marker files")
    parser.add_argument("--clip-lengths", type=int, nargs="+", default=(12, 60))
    parser.add_argument("--min-overlap-seconds", type=float, default=1.0)
    parser.add_argument("--limit-per-split", type=int,
                        help="Metadata-only sample limit for each official split; cannot establish global disjointness")
    parser.add_argument("--dry-run", action="store_true", help="Validate metadata without writing output")
    parser.add_argument("--compute-train-stats", action="store_true",
                        help="After marker generation, compute scaler pickles from preprocessed train clips only")
    parser.add_argument("--preproc-dir", type=Path,
                        help="Preprocessed FFT clip root required by --compute-train-stats")
    parser.add_argument("--stats-population", choices=("balanced", "all"), default="balanced",
                        help="Training clips used for stats; balanced mirrors loader default sampling_ratio=1")
    parser.add_argument("--stats-sampling-ratio", type=float, default=1.0)
    parser.add_argument("--stats-seed", type=int, default=123)
    args = parser.parse_args()
    if any(length <= 0 for length in args.clip_lengths) or args.min_overlap_seconds < 0:
        parser.error("clip lengths must be positive and minimum overlap must be non-negative")
    report = prepare(args.dataset_root, args.output_dir, tuple(args.clip_lengths),
                     args.min_overlap_seconds, args.limit_per_split, args.dry_run)
    if args.compute_train_stats:
        if args.dry_run:
            parser.error("--compute-train-stats cannot be combined with --dry-run")
        if args.limit_per_split is not None:
            parser.error("--compute-train-stats requires the complete official training partition")
        if args.preproc_dir is None:
            parser.error("--compute-train-stats requires --preproc-dir")
        report["train_stats_files"] = {
            str(length): compute_train_stats(args.output_dir, args.preproc_dir, length,
                                             args.stats_population, args.stats_sampling_ratio,
                                             args.stats_seed)
            for length in args.clip_lengths
        }
        (args.output_dir / "metadata.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
