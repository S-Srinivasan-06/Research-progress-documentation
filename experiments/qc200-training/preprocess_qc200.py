"""Generate or audit explicit QC200 feature rows without changing marker labels."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import numpy as np
import h5py
from scipy.fftpack import fft
from scipy.signal import resample
from run_contract import SOURCE_VERSION, load_config, write_json
from tusz_annotations import CHANNELS, channel_reference, seizure_intervals, clip_label

def log_fft(signal):
    if signal.shape != (19, 2400) or not np.isfinite(signal).all():
        raise ValueError("Expected finite (19,2400) waveform")
    seconds=signal.reshape(19,12,200).transpose(1,0,2)
    amplitude=np.abs(fft(seconds,n=200,axis=-1)[...,:100])
    amplitude[amplitude==0]=1e-8
    return np.log(amplitude).astype(np.float32)

def read_markers(root):
    rows=[]; sets={}
    for split in ("train","dev","test"):
        patients=set()
        for suffix,y in (("sz",1),("nosz",0)):
            for line in (root/f"{split}Set_seq2seq_12s_{suffix}.txt").read_text().splitlines():
                if not line.strip(): continue
                rel,label=line.rsplit(",",1); rel=rel.strip().replace("\\","/")
                if int(label)!=y or rel.startswith("/") or ".." in Path(rel).parts:
                    raise ValueError("Invalid marker identity/label")
                expected="eval" if split=="test" else split
                parts=Path(rel).parts
                if len(parts)<3 or parts[0]!=expected: raise ValueError("Marker partition mismatch")
                patients.add(parts[1]); rows.append((rel,y,split,parts[1]))
        sets[split]=patients
    if len({r[0] for r in rows})!=len(rows): raise ValueError("Duplicate marker identities")
    for a,b in (("train","dev"),("train","test"),("dev","test")):
        if sets[a]&sets[b]: raise ValueError("Patient leakage across partitions")
    return rows

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    args=parser.parse_args(); config=load_config(args.config)
    rows=read_markers(Path(config["marker_dir"]))
    mode=config.get("preprocess_mode","audit_cache")
    if mode not in ("audit_cache","generate"): raise ValueError("Invalid preprocessing mode")
    root=Path(config["preproc_dir"]); output=Path(config["output_dir"])/"preprocessing"
    output.mkdir(parents=True,exist_ok=True)
    selected=rows[:config.get("preprocess_limit",len(rows))]
    groups=defaultdict(list)
    for row in selected:
        match=re.fullmatch(r"(.+\.edf)_(\d+)\.h5",row[0])
        if not match: raise ValueError("Invalid clip path")
        groups[match[1]].append((row,int(match[2])))
    if mode=="generate":
        import pyedflib
        raw_root=Path(config["raw_edf_dir"])
        for relative,clips in groups.items():
            source=raw_root/relative
            with pyedflib.EdfReader(str(source)) as reader:
                labels=reader.getSignalLabels()
                reference,ordered=channel_reference(labels)
                normalized=[re.sub(r"\s+"," ",x.strip()).upper() for x in labels]
                indices=[normalized.index(label) for label in ordered]
                rates=[reader.getSampleFrequency(i) for i in indices]
                if len(set(rates))!=1: raise ValueError("Mixed selected-channel sample rates")
                # Decode every selected channel explicitly: no swallowed read errors.
                signals=np.stack([reader.readSignal(i) for i in indices])
                if not np.isfinite(signals).all(): raise ValueError("Nonfinite raw samples")
                if np.any(np.ptp(signals,axis=1)==0): raise ValueError("Constant required channel")
                duration=signals.shape[1]/rates[0]
                if rates[0]!=200:
                    # Match historical full-recording Fourier resampling including floor duration.
                    signals=resample(signals,int(duration)*200,axis=1)
                annotation=source.with_suffix(".csv_bi")
                if not annotation.exists(): annotation=source.with_suffix(".csv")
                events=seizure_intervals(annotation)
                for (rel,y,split,patient),index in clips:
                    start=index*12; stop=start+12
                    if clip_label(start,stop,events)!=y: raise ValueError("Marker/annotation label mismatch")
                    features=log_fft(signals[:,start*200:stop*200])
                    target=root/rel; target.parent.mkdir(parents=True,exist_ok=True)
                    if target.exists():
                        with h5py.File(target,"r") as existing:
                            if not np.array_equal(existing["clip"][()],features):
                                raise ValueError("Existing feature differs; use a new feature directory")
                        continue
                    temporary=target.with_suffix(".h5.tmp")
                    with h5py.File(temporary,"w") as result:
                        result.create_dataset("clip",data=features)
                        result.attrs["feature_version"]=config["feature_version"]
                        result.attrs["producer_version"]=SOURCE_VERSION
                        result.attrs["reference"]=reference
                    temporary.replace(target)
    reports={}; missing=[]; inventory=[]; unversioned=0
    for split in ("train","dev","test"):
        subset=[r for r in selected if r[2]==split]
        sums=np.zeros(19); squares=np.zeros(19); n=0; verified=0; extrema=[float("inf"),float("-inf")]
        for rel,y,_,patient in subset:
            path=root/rel
            if not path.is_file(): missing.append(rel); continue
            with h5py.File(path,"r") as record:
                x=np.asarray(record["clip"])
                feature_version=str(record.attrs.get("feature_version",""))
                producer_version=str(record.attrs.get("producer_version",""))
            if feature_version!=config["feature_version"] or producer_version!=SOURCE_VERSION:
                unversioned+=1
            inventory.append({"row":rel,"bytes":path.stat().st_size,"mtime_ns":path.stat().st_mtime_ns,
                              "feature_version":feature_version,"producer_version":producer_version})
            if x.shape!=(12,19,100) or x.dtype!=np.float32 or not np.isfinite(x).all():
                raise ValueError("Invalid feature shape/type/values")
            values=x.astype(np.float64)
            sums+=values.sum(axis=(0,2)); squares+=(values*values).sum(axis=(0,2)); n+=1200
            extrema=[min(extrema[0],float(x.min())),max(extrema[1],float(x.max()))]; verified+=1
        reports[split]={"rows":len(subset),"positive":sum(r[1] for r in subset),
                        "patients":len({r[3] for r in subset}),"verified_features":verified,
                        "mean":(sums/n).tolist() if n else None,
                        "std":np.sqrt(np.maximum(squares/n-(sums/n)**2,0)).tolist() if n else None,
                        "range":extrema if n else None}
    write_json(output/"missing_features.json",missing)
    write_json(output/"validated_marker_rows.json",[list(row) for row in selected])
    write_json(output/"feature_inventory.json",inventory)
    legacy_approved=False
    if unversioned and config.get("legacy_provenance_record"):
        provenance=json.loads(Path(config["legacy_provenance_record"]).read_text())
        evidence=[Path(p) for p in provenance.get("evidence_files",[])]
        legacy_approved=(provenance.get("feature_version")==config["feature_version"]
                         and provenance.get("approved") is True
                         and provenance.get("reviewed_by")
                         and provenance.get("source_cache_agreement_verified") is True
                         and bool(evidence) and all(p.is_file() for p in evidence))
        write_json(output/"legacy_provenance_record.json",provenance)
    report={"mode":mode,"feature_root":str(root.resolve()),"feature_version":config["feature_version"],"selected":len(selected),
            "expected":len(rows),"full_corpus":len(selected)==len(rows),"missing":len(missing),
            "shape":[12,19,100],"channels":list(CHANNELS),"sample_rate_hz":200,
            "fft":"scipy.fftpack, 200 samples, bins 0..99, natural log; zero amplitude->1e-8",
            "splits":reports,"unversioned_features":unversioned,
            "provenance_approved":bool(config.get("feature_contract_approved",False)
                                       and (unversioned==0 or legacy_approved)),
            "inventory_limit":"File identities, sizes and producer metadata; no content fingerprint"}
    write_json(output/"feature_validation.json",report)
    if missing: raise RuntimeError(f"Missing {len(missing)} features; details saved privately")
    print(json.dumps({"verified":len(selected),"full_corpus":report["full_corpus"]}))

if __name__=="__main__": main()
