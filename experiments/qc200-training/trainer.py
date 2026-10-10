#!/usr/bin/env python3
"""Private 200 Hz DGDCN training runner for staged TUSZ feature clips.

Inputs: preproc_dir/{train,dev,eval}/<marker path>.h5 with dataset `clip`
and markers named train/dev/testSet_seq2seq_12s_{sz,nosz}.txt. Evaluation
partition is named `eval` on disk and `test` in marker files.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import pickle
import random
import sys
import time
from collections import Counter
from pathlib import Path, PurePosixPath

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, confusion_matrix,
                             precision_recall_curve, average_precision_score)
from threshold_policy import make_threshold_policy, select_threshold as choose_threshold, ensure_resume_compatible
from run_contract import SOURCE_VERSION, write_json, rows_contract, sampler_statistics


def marker_rows(marker_dir: Path, split: str):
    rows = []
    for kind, label in (("sz", 1), ("nosz", 0)):
        path = marker_dir / f"{split}Set_seq2seq_12s_{kind}.txt"
        if not path.is_file():
            raise FileNotFoundError(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rel, raw_label = line.rsplit(",", 1)
            rel = PurePosixPath(rel.strip().replace("\\", "/"))
            if rel.is_absolute() or any(p in ("", ".", "..") for p in rel.parts):
                raise ValueError(f"Unsafe marker path: {rel}")
            y = int(raw_label)
            if y != label:
                raise ValueError(f"Label/file mismatch in {path}: {line}")
            rows.append((rel.as_posix(), y, rel.parts[1] if len(rel.parts) > 2 else rel.parts[0]))
    if not rows:
        raise ValueError(f"No examples in {split} marker files")
    return rows


class Clips(Dataset):
    def __init__(self, root: Path, rows, mean, std, augment=False):
        self.root, self.rows = root, rows
        self.mean, self.std = mean.astype(np.float32), std.astype(np.float32)
        self.augment = augment
        # Channel reflection pairs copied from the repaired data.data_utils.get_swap_pairs
        # implementation; this avoids importing its pyedflib dependency in a cached-feature run.
        channels = ["EEG FP1", "EEG FP2", "EEG F3", "EEG F4", "EEG C3", "EEG C4",
                    "EEG P3", "EEG P4", "EEG O1", "EEG O2", "EEG F7", "EEG F8",
                    "EEG T3", "EEG T4", "EEG T5", "EEG T6", "EEG FZ", "EEG CZ", "EEG PZ"]
        self.swap_pairs = [(channels.index(a), channels.index(b)) for a,b in
            (("EEG FP1","EEG FP2"),("EEG F3","EEG F4"),("EEG F7","EEG F8"),
             ("EEG C3","EEG C4"),("EEG T3","EEG T4"),("EEG T5","EEG T6"),("EEG O1","EEG O2"))]
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        rel, y, _ = self.rows[i]
        path = self.root / rel
        with h5py.File(path, "r") as f:
            x = f["clip"][()]
        if x.shape != (12, 19, 100):
            raise ValueError(f"Expected clip (12,19,100), got {x.shape} at {path}")
        x = x.astype(np.float32, copy=False)
        if not np.isfinite(x).all():
            raise ValueError(f"Nonfinite feature values: {path}")
        if self.augment:
            if random.choice((True, False)):
                for left, right in self.swap_pairs:
                    x[:, [left, right], :] = x[:, [right, left], :]
            x += np.log(random.uniform(0.8, 1.2))
        x = (x - self.mean) / self.std
        return torch.from_numpy(np.ascontiguousarray(x)), torch.tensor(float(y)), rel


def balanced_train_rows(rows, seed: int):
    positives = [r for r in rows if r[1] == 1]
    negatives = [r for r in rows if r[1] == 0]
    if not positives or len(negatives) < len(positives):
        raise ValueError("Need positive training clips and at least as many negatives for balanced normalization")
    rng = np.random.RandomState(seed)
    chosen_neg = [negatives[i] for i in rng.permutation(len(negatives))[:len(positives)]]
    selected = positives + chosen_neg
    rng.shuffle(selected)
    return selected


def scaler_from_balanced_training(root: Path, selected, deadline=None, reserve_seconds=600):
    """Compute per-channel moments over the exact balanced training pool."""
    sums = np.zeros(19, np.float64); squares = np.zeros(19, np.float64); n = 0
    print(f"Computing normalization from {len(selected):,} balanced training clips...",flush=True)
    for j, (rel, _, _) in enumerate(selected, 1):
        if deadline is not None and deadline - time.monotonic() <= reserve_seconds:
            raise TimeoutError("Runtime reserve reached during normalization; no training epoch started.")
        with h5py.File(root / rel, "r") as f:
            x = np.asarray(f["clip"], dtype=np.float64)
        if x.shape != (12, 19, 100) or not np.isfinite(x).all():
            raise ValueError(f"Invalid training feature values/shape: {rel}")
        sums += x.sum(axis=(0, 2)); squares += np.square(x).sum(axis=(0, 2)); n += 12 * 100
        if j % 1000 == 0: print(f"normalization clips={j:,}/{len(selected):,}",flush=True)
    mean = sums / n
    std = np.sqrt(np.maximum(squares / n - mean * mean, 1e-12))
    print(f"Normalization complete: {len(selected):,} clips",flush=True)
    return mean.reshape(1, 19, 1), std.reshape(1, 19, 1), sum(r[1] for r in selected), sum(1-r[1] for r in selected)


def metrics(y, p, threshold=0.5):
    pred = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    try: auc = float(roc_auc_score(y, p))
    except ValueError: auc = float("nan")
    try: ap = float(average_precision_score(y, p))
    except ValueError: ap = float("nan")
    return {"loss": None, "accuracy": float(accuracy_score(y, pred)), "f1": float(f1_score(y, pred, zero_division=0)),
            "precision": float(precision_score(y, pred, zero_division=0)), "recall": float(recall_score(y, pred, zero_division=0)),
            "specificity": float(tn / max(tn + fp, 1)), "auroc": auc, "average_precision": ap, "threshold": threshold,
            "false_positive_rate": float(fp/max(tn+fp,1)),
            "tp":int(tp), "fn":int(fn), "tn":int(tn), "fp":int(fp),
            "count": int(len(y)), "positive_count": int(np.sum(y)), "negative_count": int(len(y)-np.sum(y))}


def evaluate(model, loader, device, loss_fn, threshold=0.5, select_threshold=False,
             prediction_csv_gz=None, threshold_objective="f1", target_sensitivity=0.95):
    model.eval(); ys=[]; ps=[]; losses=[]
    with torch.no_grad():
        for x, y, _ in loader:
            x=x.to(device); y=y.to(device)
            logits=binary_logit(model(x))
            losses.append(float(loss_fn(logits, y).item()) * len(y))
            ps.extend(torch.sigmoid(logits).cpu().numpy().tolist()); ys.extend(y.cpu().numpy().astype(int).tolist())
    y=np.asarray(ys); p=np.asarray(ps)
    if select_threshold:
        threshold=choose_threshold(y,p,threshold_objective,target_sensitivity)
    m=metrics(y, p, threshold)
    m["loss"] = sum(losses)/len(ys)
    if select_threshold:
        m["threshold_objective"] = threshold_objective
        m["target_sensitivity"] = target_sensitivity if threshold_objective == "sensitivity" else None
        m["achieved_sensitivity"] = m["recall"]
    if prediction_csv_gz is not None:
        Path(prediction_csv_gz).parent.mkdir(parents=True,exist_ok=True)
        with gzip.open(prediction_csv_gz,"wt",encoding="utf-8",newline="") as stream:
            writer=csv.writer(stream); writer.writerow(("relpath","y_true","probability"))
            # Preserve original marker order from the non-shuffled evaluation dataset.
            for (rel,truth,_),prob in zip(loader.dataset.rows,p): writer.writerow((rel,truth,float(prob)))
    return m


def binary_logit(output):
    if output.ndim == 1:
        return output
    if output.ndim == 2 and output.shape[1] == 1:
        return output[:, 0]
    raise ValueError(f"Expected scalar binary logits shaped (B,) or (B,1), got {tuple(output.shape)}")


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def save_checkpoint(path, model, optimizer, scheduler, epoch, best, config, progress=None):
    tmp=path.with_suffix(path.suffix+".tmp")
    with np.load(path.parent/"train_normalization.npz") as normalization:
        mean=normalization["mean"].copy(); std=normalization["std"].copy()
    torch.save({"epoch":epoch,"model":model.state_dict(),"optimizer":optimizer.state_dict(),
                "scheduler":scheduler.state_dict(),"best_dev":best,"rng":rng_state(),"config":config,
                "threshold_policy":config.get("threshold_policy"),
                "normalization_mean":mean,"normalization_std":std,
                "progress":progress or {}},tmp)
    os.replace(tmp,path)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo",type=Path,required=True,help="DGDCN reproduction source root")
    ap.add_argument("--preproc-dir",type=Path,required=True,help="Extracted feature cache")
    ap.add_argument("--marker-dir",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--adjacency",type=Path,default=None)
    ap.add_argument("--epochs",type=int,default=100); ap.add_argument("--batch-size",type=int,default=20)
    ap.add_argument("--eval-batch-size",type=int,default=128)
    ap.add_argument("--workers",type=int,default=4); ap.add_argument("--lr",type=float,default=1e-4)
    ap.add_argument("--weight-decay",type=float,default=5e-4); ap.add_argument("--seed",type=int,default=123)
    ap.add_argument("--max-runtime-hours",type=float,default=11.5,help="Pause before this runtime; resume from last.pt")
    ap.add_argument("--save-reserve-minutes",type=float,default=10,help="Reserve runtime for final metrics/output flush")
    ap.add_argument("--patient-balanced-sampler",action="store_true",help="Explicitly weight each training patient's total sampling mass equally")
    ap.add_argument("--threshold-objective",choices=("f1","sensitivity"),default="f1",
                    help="Select the dev threshold by F1 (default) or target sensitivity")
    ap.add_argument("--target-sensitivity",type=float,default=0.95,
                    help="Minimum empirical dev recall when --threshold-objective=sensitivity (0,1]")
    ap.add_argument("--resume",type=Path,default=None); ap.add_argument("--normalization",type=Path,default=None)
    ap.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--feature-version", required=True)
    ap.add_argument("--variant", choices=("B1","W1"), default="B1")
    ap.add_argument("--pause-file",type=Path,default=None)
    args=ap.parse_args()
    if args.resume and args.resume.resolve()!=(args.output_dir/"last.pt").resolve():
        ap.error("Resume must use this run's explicit last.pt, never best.pt or another run")
    if args.normalization is not None:
        ap.error("QC200 computes its own normalization; shared artifacts need a separate audited import")
    if not args.patient_balanced_sampler:
        ap.error("QC200 requires patient-total-equal sampling")
    if args.output_dir.exists() and not args.resume and any(
            p.name != "preprocessing" for p in args.output_dir.iterdir()):
        ap.error("Fresh experiments require an empty output directory")
    if args.epochs < 1 or args.batch_size < 2 or args.eval_batch_size < 1 or args.max_runtime_hours <= 0 or args.save_reserve_minutes < 0: ap.error("invalid epoch, batch-size, or runtime setting")
    try:
        threshold_policy=make_threshold_policy(args.threshold_objective,args.target_sensitivity)
    except ValueError as exc:
        ap.error(str(exc))
    if threshold_policy["objective"] == "sensitivity" and args.resume is None:
        saved_run_files=("config.json","last.pt","best.pt","metrics.jsonl","final_eval_metrics.json")
        if any((args.output_dir/name).is_file() for name in saved_run_files):
            ap.error("Sensitivity selection cannot reuse an output directory with saved run artifacts; use a new output directory or resume with its saved policy")
    print(f"Dev threshold policy: objective={threshold_policy['objective']} target_sensitivity={threshold_policy['target_sensitivity']}",flush=True)
    repo=args.repo.resolve(); sys.path.insert(0,str(repo))
    from model.DGDCN.model.DGDCN_r import make_model_2
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available(): raise RuntimeError("CUDA selected but no CUDA device is available.")
        print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}",flush=True)
    elif args.device != "cpu":
        raise ValueError("--device must be cpu or cuda[:index]")
    else:
        print("CPU device selected.",flush=True)
    torch.backends.cudnn.enabled=False  # preserve released training semantics
    deadline=time.monotonic()+args.max_runtime_hours*3600
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    train_all=marker_rows(args.marker_dir,"train"); dev=marker_rows(args.marker_dir,"dev"); test=marker_rows(args.marker_dir,"test")
    patient_sets={k:{r[2] for r in v} for k,v in (("train",train_all),("dev",dev),("eval",test))}
    for a,b in (("train","dev"),("train","eval"),("dev","eval")):
        overlap=patient_sets[a]&patient_sets[b]
        if overlap: raise ValueError(f"Patient overlap {a}/{b}: {len(overlap)} IDs; examples={sorted(overlap)[:5]}")
    train=balanced_train_rows(train_all,args.seed)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    summary=sampler_statistics(train)
    write_json(args.output_dir/"environment.json",{
        "python":sys.version,"torch":torch.__version__,"numpy":np.__version__,
        "h5py":h5py.__version__,"device":args.device})
    identity={"source_version":SOURCE_VERSION, "run_id":args.run_id,
              "feature_root":str(args.preproc_dir.resolve()),
              "feature_version":args.feature_version, "variant":args.variant,
              "seed":args.seed, "epochs":args.epochs, "batch_size":args.batch_size,
              "lr":args.lr, "weight_decay":args.weight_decay,
              "sampler":"patient-total-equal", "normalization":"balanced-pool-channelwise",
              "threshold_policy":threshold_policy, "augmentation":"reflection-logscale",
              "model":"released-submodule1-max-two-logits-fixed-cheb-attention"}
    # Exact identities and normalization arrays are compared on resume, without hashing.
    identity_path=args.output_dir/"run_identity.json"
    row_path=args.output_dir/"training_pool.json"
    partitions={k:rows_contract(v) for k,v in (("train",train_all),("dev",dev),("eval",test))}
    if args.resume:
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError("Resume configuration/source version differs from saved experiment")
        if json.loads(row_path.read_text()) != rows_contract(train):
            raise ValueError("Resume training pool changed")
        if json.loads((args.output_dir/"partition_manifest.json").read_text()) != partitions:
            raise ValueError("Resume partition identities changed")
    else:
        write_json(identity_path,identity)
        write_json(row_path,rows_contract(train))
        write_json(args.output_dir/"partition_manifest.json",partitions)
        write_json(args.output_dir/"sampler_summary.json",summary)
        # Source snapshots support exact comparison on resume, including the model helper.
    import shutil
    source_files=list(Path(__file__).parent.glob("*.py"))+list((repo/"model").rglob("*.py"))
    for source in source_files:
        rel=source.relative_to(Path(__file__).parent) if source.is_relative_to(Path(__file__).parent) else Path("repo")/source.relative_to(repo)
        saved_source=args.output_dir/"source_snapshot"/rel
        if args.resume:
            if not saved_source.is_file() or source.read_bytes()!=saved_source.read_bytes():
                raise ValueError(f"Resume source changed: {rel}")
        else:
            saved_source.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,saved_source)
    adj_source=args.adjacency or repo/"data/electrode_graph/adj_mx_3d.pkl"
    saved_adj=args.output_dir/"source_snapshot"/"adj_mx_3d.pkl"
    if args.resume:
        if not saved_adj.is_file() or adj_source.read_bytes()!=saved_adj.read_bytes():
            raise ValueError("Resume graph adjacency changed")
    else:
        shutil.copyfile(adj_source,saved_adj)
    normalization_path=args.normalization or args.output_dir/"train_normalization.npz"
    if args.resume:
        if not args.resume.is_file() or not normalization_path.is_file():
            raise FileNotFoundError("Resume requires both the explicit checkpoint and saved normalization artifact")
        with np.load(normalization_path) as saved:
            mean,std=saved["mean"],saved["std"]
            npos,nneg=int(saved["positive_clips"]),int(saved["negative_clips"])
        if mean.shape != (1,19,1) or std.shape != (1,19,1) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
            raise ValueError("Saved normalization artifact has invalid shape or values")
    else:
        mean,std,npos,nneg=scaler_from_balanced_training(args.preproc_dir,train,deadline,args.save_reserve_minutes*60)
        np.savez(normalization_path,mean=mean,std=std,positive_clips=npos,negative_clips=nneg,seed=args.seed)
        write_json(args.output_dir/"normalization_provenance.json",{
            "feature_version":args.feature_version,"pool":"training_pool.json",
            "pre_augmentation":True,"shape":list(mean.shape),"positive_clips":npos,
            "negative_clips":nneg,"mean":mean.tolist(),"std":std.tolist()})
    train_ds=Clips(args.preproc_dir,train,mean,std,augment=True); dev_ds=Clips(args.preproc_dir,dev,mean,std)
    sampler=None; shuffle=True
    if args.patient_balanced_sampler:
        counts=Counter(r[2] for r in train); weights=[1.0/counts[r[2]] for r in train]
        sampler=WeightedRandomSampler(torch.DoubleTensor(weights),len(weights),replacement=True); shuffle=False
    tl=DataLoader(train_ds,batch_size=args.batch_size,shuffle=shuffle,sampler=sampler,num_workers=args.workers,pin_memory=True,drop_last=True)
    dl=DataLoader(dev_ds,batch_size=args.eval_batch_size,shuffle=False,num_workers=args.workers,pin_memory=True)
    adjacency=args.adjacency or repo/"data/electrode_graph/adj_mx_3d.pkl"
    with adjacency.open("rb") as f: adj=pickle.load(f)[2]
    device=torch.device(args.device)
    model=make_model_2(nb_block=4,in_channels=100,K=3,nb_chev_filter=64,nb_time_filter=64,time_strides=1,
        adj_mx=adj,num_for_predict=12,len_input=12,num_of_vertices=19,DEVICE=device).to(device)
    model.eval()
    smoke_x=torch.zeros((2,12,19,100),device=device,requires_grad=True)
    smoke_logits=binary_logit(model(smoke_x))
    if tuple(smoke_logits.shape)!=(2,): raise RuntimeError(f"Model must return one scalar per example; got {tuple(smoke_logits.shape)}")
    smoke_logits.sum().backward()
    if smoke_x.grad is None or not torch.isfinite(smoke_x.grad).all(): raise RuntimeError("DGDCN input gradient smoke check failed")
    model.zero_grad(set_to_none=True); model.train()
    print("Model startup check: output=(B,), finite input gradient",flush=True)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)
    pos_weight=1.0 if args.variant=="B1" else summary["candidate_pos_weight"]
    loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight,device=device))
    dev_loss_fn=torch.nn.BCEWithLogitsLoss()
    start=0; best=-float("inf"); best_threshold=0.5
    patience_count=0; best_dev_loss=float("inf")
    def split_counts(rows):
        return {"clips":len(rows),"positive":sum(r[1] for r in rows),"negative":sum(1-r[1] for r in rows),"patients":len({r[2] for r in rows})}
    config={"input_shape":[12,19,100],"sample_rate_hz":200,"feature_bins":100,"normalization":"per-channel moments over seeded balanced training pool","normalization_clips":{"positive":npos,"negative":nneg},"training_clip_policy":"all train positives plus seed-sampled equal-count negatives","augmentation":{"enabled":True,"method":"reflection plus FFT log-scale augmentation from repaired loader"},"sampler":"patient-equal-total-mass replacement sampler on balanced pool" if sampler else "uniform shuffle on balanced pool","official_partitions":{"train":"train","dev":"dev","eval":"test markers"},"partition_counts":{"train":split_counts(train),"train_before_undersampling":split_counts(train_all),"dev":split_counts(dev),"eval":split_counts(test)},"head":"repaired model returns scalar binary logit; BCEWithLogitsLoss","threshold_policy":threshold_policy,"args":vars(args)}
    config.update({"identity":identity,"training_pos_weight":pos_weight,"dev_loss":"unweighted BCE",
                   "head":"released Linear(64,2) reduced by maximum to binary logit",
                   "eval_on_completion":False})
    if args.resume:
        ck=torch.load(args.resume,map_location=device,weights_only=False); model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"]); scheduler.load_state_dict(ck["scheduler"])
        if ck.get("config",{}).get("identity") != identity:
            raise ValueError("Checkpoint identity does not match this experiment")
        if not np.array_equal(ck.get("normalization_mean"),mean) or not np.array_equal(ck.get("normalization_std"),std):
            raise ValueError("Checkpoint normalization differs from saved artifact")
        saved_config=ck.get("config",config)
        checkpoint_policy=ck.get("threshold_policy")
        config_policy=saved_config.get("threshold_policy") if isinstance(saved_config,dict) else None
        if checkpoint_policy is not None and config_policy is not None and checkpoint_policy != config_policy:
            raise ValueError("Resume checkpoint has conflicting threshold policy metadata")
        policy_in_checkpoint=checkpoint_policy if checkpoint_policy is not None else config_policy
        saved_policy=ensure_resume_compatible(policy_in_checkpoint,threshold_policy)
        run_config_path=args.output_dir/"config.json"
        if run_config_path.is_file():
            run_config=json.loads(run_config_path.read_text(encoding="utf-8"))
            run_policy=run_config.get("threshold_policy")
            # A config without policy metadata belongs to a historical F1 run.
            ensure_resume_compatible(run_policy,saved_policy)
        config=dict(saved_config)
        config["threshold_policy"]=saved_policy
        # Both checkpoint forms store the number of completed epochs. A partial
        # epoch is replayed at that index; a boundary checkpoint advances from it.
        start=int(ck["epoch"])
        if start < 0 or start > args.epochs: raise ValueError("Resume checkpoint has an invalid completed-epoch position")
        best=float(ck["best_dev"]["score"]); best_threshold=float(ck["best_dev"]["threshold"])
        patience_count=int(ck.get("progress",{}).get("patience_count",0)); best_dev_loss=float(ck.get("progress",{}).get("best_dev_loss",float("inf")))
        random.setstate(ck["rng"]["python"]); np.random.set_state(ck["rng"]["numpy"]); torch.set_rng_state(ck["rng"]["torch"].cpu())
        if torch.cuda.is_available() and ck["rng"]["cuda"] is not None: torch.cuda.set_rng_state_all([s.cpu() for s in ck["rng"]["cuda"]])
    early_stopped = patience_count >= 5
    if not args.resume or not (args.output_dir/"config.json").is_file():
        (args.output_dir/"config.json").write_text(json.dumps(config,indent=2,default=str)+"\n",encoding="utf-8")
    history=args.output_dir/"metrics.jsonl"
    last_checkpoint=time.monotonic(); paused=False
    for epoch in (() if early_stopped else range(start,args.epochs)):
        model.train(); running=0.0; seen=0
        for step,(x,y,_) in enumerate(tl,1):
            if (args.pause_file and args.pause_file.exists()) or deadline-time.monotonic() <= args.save_reserve_minutes*60:
                save_checkpoint(args.output_dir/"last.pt",model,optimizer,scheduler,epoch,{"score":best,"threshold":best_threshold},config,{"patience_count":patience_count,"best_dev_loss":best_dev_loss,"restart_epoch":True,"paused":True})
                (args.output_dir/"run-status.json").write_text(json.dumps({"state":"paused_runtime","completed_epochs":epoch,"resume":"last.pt restarts the partial epoch"},indent=2)+"\n")
                print("Runtime reserve reached; saved checkpoint; eval partition was not read.",flush=True); paused=True; break
            x=x.to(device); y=y.to(device); optimizer.zero_grad(set_to_none=True); logits=binary_logit(model(x)); loss=loss_fn(logits,y)
            if not torch.isfinite(loss): raise RuntimeError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0,error_if_nonfinite=True)
            optimizer.step(); running+=loss.item()*len(y); seen+=len(y)
            if step==1:
                print(json.dumps({"state":"optimization_started","epoch":epoch+1,"step":step,
                                  "device":str(device),"loss":float(loss.item()),"pos_weight":pos_weight}),flush=True)
            if time.monotonic()-last_checkpoint >= 600:
                save_checkpoint(args.output_dir/"last.pt",model,optimizer,scheduler,epoch,{"score":best,"threshold":best_threshold},config,{"patience_count":patience_count,"best_dev_loss":best_dev_loss,"restart_epoch":True,"paused":False})
                print(f"epoch={epoch+1} step={step}/{len(tl)} train_loss={running/max(seen,1):.5f} checkpoint=last.pt",flush=True); last_checkpoint=time.monotonic()
        if paused: break
        scheduler.step()
        vm=evaluate(model,dl,device,dev_loss_fn,select_threshold=True,
                    threshold_objective=threshold_policy["objective"],
                    target_sensitivity=args.target_sensitivity)
        score=vm["auroc"]
        if not np.isfinite(score): score=-vm["loss"]
        improved=score>best
        if improved: best=score; best_threshold=vm["threshold"]
        if vm["loss"] < best_dev_loss: best_dev_loss=vm["loss"]; patience_count=0
        else: patience_count+=1
        row={"epoch":epoch+1,"train_loss":running/max(seen,1),"dev":vm,"lr":optimizer.param_groups[0]["lr"],"best":improved}
        with history.open("a",encoding="utf-8") as f: f.write(json.dumps(row,allow_nan=True)+"\n")
        progress={"patience_count":patience_count,"best_dev_loss":best_dev_loss,"restart_epoch":False,"paused":False}
        save_checkpoint(args.output_dir/"last.pt",model,optimizer,scheduler,epoch+1,{"score":best,"threshold":best_threshold},config,progress)
        if improved: save_checkpoint(args.output_dir/"best.pt",model,optimizer,scheduler,epoch+1,{"score":best,"threshold":best_threshold},config,progress)
        print(json.dumps(row,allow_nan=True),flush=True)
        last_checkpoint=time.monotonic()
        if patience_count >= 5: early_stopped=True; break
    if paused: return 0
    if not (args.output_dir/"best.pt").exists(): raise RuntimeError("No dev-selected checkpoint exists; eval partition will not be opened.")
    best_ck=torch.load(args.output_dir/"best.pt",map_location=device,weights_only=False)
    model.load_state_dict(best_ck["model"])
    dev_final=evaluate(model,dl,device,dev_loss_fn,select_threshold=True,
                       threshold_objective=threshold_policy["objective"],
                       target_sensitivity=args.target_sensitivity,
                       prediction_csv_gz=args.output_dir/"dev_predictions.csv.gz")
    write_json(args.output_dir/"dev_metrics.json",dev_final)
    write_json(args.output_dir/"run-status.json",{"state":"complete_train_dev","early_stopped":early_stopped,
               "eval_features_or_scores_accessed":False,"eval_markers_checked_for_partition_identity":True,
               "best_epoch":best_ck["epoch"]})
    return 0

if __name__=="__main__": main()
