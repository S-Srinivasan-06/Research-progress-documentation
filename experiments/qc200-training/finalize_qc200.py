"""Explicit final evaluation after a development selection record is frozen."""
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
from run_contract import load_config, write_json, verify_feature_inventory

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True,type=Path)
    args=parser.parse_args(); config=load_config(args.config)
    if not config.get("selection_record"):
        raise ValueError("Final evaluation requires an explicit frozen selection_record")
    record_path=Path(config["selection_record"])
    selection=json.loads(record_path.read_text())
    out=Path(config["output_dir"])
    status=json.loads((out/"run-status.json").read_text())
    if status["state"]!="complete_train_dev": raise ValueError("Training/dev must complete first")
    if selection.get("run_id")!=config["run_id"] or selection.get("selected_on")!="dev":
        raise ValueError("Selection record must identify this run and development-only selection")
    if not selection.get("frozen_at_utc") or not selection.get("previous_eval_access_disclosed"):
        raise ValueError("Selection must be frozen and disclose historical evaluation access")
    if (out/"eval_access_log.jsonl").exists():
        raise ValueError("Evaluation already attempted; inspect the recorded access before any retry")
    import sys
    import torch
    import numpy as np
    from torch.utils.data import DataLoader
    import trainer
    sys.path.insert(0,str(Path(__file__).parent/"repo"))
    from model.DGDCN.model.DGDCN_r import make_model_2
    import pickle
    checkpoint=torch.load(out/"best.pt",map_location="cpu",weights_only=False)
    if checkpoint["epoch"]!=selection.get("checkpoint_epoch"):
        raise ValueError("Selected checkpoint epoch differs")
    threshold=float(selection["threshold"])
    if not 0<=threshold<=1: raise ValueError("Invalid probability threshold")
    dev=json.loads((out/"dev_metrics.json").read_text())
    if threshold!=dev["threshold"]:
        raise ValueError("Selection threshold differs from saved development selection")
    rows=trainer.marker_rows(Path(config["marker_dir"]),"test")
    saved_rows=json.loads((out/"partition_manifest.json").read_text())["eval"]
    if saved_rows!=[list(row) for row in rows]: raise ValueError("Evaluation marker identities changed")
    identity=json.loads((out/"run_identity.json").read_text())
    if identity["run_id"]!=config["run_id"] or identity["variant"]!=config.get("variant","B1"):
        raise ValueError("Configuration does not identify this saved run")
    if identity["feature_root"]!=str(Path(config["preproc_dir"]).resolve()):
        raise ValueError("Feature cache differs from training")
    verify_feature_inventory(config,out/"preprocessing")
    source_root=Path(__file__).parent
    sources=list(source_root.glob("*.py"))+list((source_root/"repo/model").rglob("*.py"))
    for source in sources:
        relative=source.relative_to(source_root)
        saved=out/"source_snapshot"/relative
        if not saved.is_file() or saved.read_bytes()!=source.read_bytes():
            raise ValueError(f"Source differs from training snapshot: {relative}")
    adjacency_path=source_root/"repo/data/electrode_graph/adj_mx_3d.pkl"
    if adjacency_path.read_bytes()!=(out/"source_snapshot/adj_mx_3d.pkl").read_bytes():
        raise ValueError("Adjacency differs from training snapshot")
    if checkpoint["config"]["identity"]!=identity or identity["feature_version"]!=config["feature_version"]:
        raise ValueError("Checkpoint/feature identity mismatch")
    with np.load(out/"train_normalization.npz") as normalization:
        mean=normalization["mean"].copy(); std=normalization["std"].copy()
    if not np.array_equal(mean,checkpoint["normalization_mean"]) or not np.array_equal(std,checkpoint["normalization_std"]):
        raise ValueError("Checkpoint normalization differs")
    device=torch.device(config.get("device","cpu"))
    with (Path(__file__).parent/"repo/data/electrode_graph/adj_mx_3d.pkl").open("rb") as f:
        adjacency=pickle.load(f)[2]
    model=make_model_2(nb_block=4,in_channels=100,K=3,nb_chev_filter=64,nb_time_filter=64,
        time_strides=1,adj_mx=adjacency,num_for_predict=12,len_input=12,num_of_vertices=19,DEVICE=device).to(device)
    model.load_state_dict(checkpoint["model"])
    dataset=trainer.Clips(Path(config["preproc_dir"]),rows,mean,std)
    loader=DataLoader(dataset,batch_size=config.get("eval_batch_size",128),shuffle=False,
                      num_workers=config.get("workers",2))
    access={"utc":datetime.now(timezone.utc).isoformat(),"run_id":config["run_id"],
            "checkpoint_epoch":checkpoint["epoch"],"threshold":threshold,
            "previously_inspected_benchmark":True}
    with (out/"eval_access_log.jsonl").open("x") as log: log.write(json.dumps(access)+"\n")
    write_json(out/"frozen_selection_record.json",selection)
    metrics=trainer.evaluate(model,loader,device,torch.nn.BCEWithLogitsLoss(),threshold=threshold,
                              prediction_csv_gz=out/"final_eval_predictions.csv.gz")
    metrics.update({"partition":"eval","previously_inspected_benchmark":True,"event_metrics":None,
                    "event_metrics_reason":"Continuous interval mapping/scoring protocol pending"})
    write_json(out/"final_eval_metrics.json",metrics)
    print(json.dumps(metrics))

if __name__=="__main__": main()
