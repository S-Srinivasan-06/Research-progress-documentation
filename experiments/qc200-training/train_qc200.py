"""Run a new QC200 training experiment; completion uses train/dev only."""
import argparse
import sys
import json
from pathlib import Path
from run_contract import load_config, verify_feature_inventory

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if not config.get("feature_contract_approved", False):
        raise ValueError("Approve feature provenance and validation before training")
    from preprocess_qc200 import read_markers
    validation_root=Path(config["output_dir"])/"preprocessing"
    report=json.loads((validation_root/"feature_validation.json").read_text())
    if not report["full_corpus"] or report["missing"] or report["feature_version"]!=config["feature_version"] or not report["provenance_approved"]:
        raise ValueError("Full, matching feature validation is required before training")
    expected=config.get("expected_partition_rows",{"train":261865,"dev":73387,"test":38384})
    frozen={"train":261865,"dev":73387,"test":38384}
    if set(expected)!=set(frozen): raise ValueError("All three partition counts must be specified")
    if expected!=frozen and not (config.get("cohort_kind")=="synthetic-test" and config["feature_version"].startswith("synthetic-")):
        raise ValueError("This QC200 version requires the frozen full cohort")
    if {k:report["splits"][k]["rows"] for k in expected}!=expected:
        raise ValueError("Feature cohort counts do not match the frozen expected partitions")
    verify_feature_inventory(config,validation_root)
    validated=json.loads((validation_root/"validated_marker_rows.json").read_text())
    if validated!=[list(row) for row in read_markers(Path(config["marker_dir"]))]:
        raise ValueError("Markers changed after feature validation")
    import trainer
    argv = ["train_qc200", "--repo", str(Path(__file__).parent / "repo"),
            "--preproc-dir", config["preproc_dir"], "--marker-dir", config["marker_dir"],
            "--output-dir", config["output_dir"], "--patient-balanced-sampler",
            "--threshold-objective", "sensitivity", "--target-sensitivity", "0.95",
            "--run-id", config["run_id"], "--feature-version", config["feature_version"],
            "--variant", config.get("variant", "B1")]
    fields = {"epochs":100, "batch-size":20, "eval-batch-size":128, "workers":2,
              "lr":1e-4, "weight-decay":5e-4, "seed":123,
              "max-runtime-hours":10, "save-reserve-minutes":10}
    for flag, default in fields.items():
        argv += ["--" + flag, str(config.get(flag.replace("-", "_"), default))]
    for key in ("resume", "device", "pause_file"):
        if config.get(key):
            argv += ["--"+key.replace("_","-"), str(config[key])]
    sys.argv = argv
    return trainer.main()

if __name__ == "__main__":
    main()
