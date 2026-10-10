"""Restore, verify and train B1, with authenticated Drive checkpoint persistence."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import filecmp
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import numpy as np
import h5py
from run_contract import load_config, write_json

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    args=parser.parse_args(); config=load_config(args.config)
    root=Path(__file__).parent; output=Path(config["output_dir"])
    control=Path(config["pipeline_control_dir"]); control.mkdir(parents=True,exist_ok=True)
    if output.exists() and any(p.name!="preprocessing" for p in output.iterdir()):
        raise ValueError("Pipeline requires a fresh B1 output; resume is a separate explicit action")
    from drive_io import drive_client,find_or_create_folder,list_children,upload_replace,signature,make_stable_snapshot
    from googleapiclient.http import MediaIoBaseDownload
    service=drive_client()
    parent=config["drive_parent_folder_id"]
    meta=service.files().get(fileId=parent,fields="id,name,mimeType,trashed,capabilities(canAddChildren)").execute()
    if meta.get("trashed") or meta.get("mimeType")!="application/vnd.google-apps.folder" or not meta.get("capabilities",{}).get("canAddChildren"):
        raise ValueError("Approved persistence folder is not writable")
    if list_children(service,parent,config["run_id"]):
        raise ValueError("Remote run name already exists; choose a new explicit run ID")
    remote=find_or_create_folder(service,parent,config["run_id"])
    remote_dirs={".":remote}; previous={}; lock=threading.Lock(); stopped=threading.Event()
    temporary=Path(tempfile.mkdtemp(prefix="qc200-sync-"))
    def publish_state(phase,**extra):
        write_json(control/"pipeline-status.json",{"utc":datetime.now(timezone.utc).isoformat(),
                   "phase":phase,"run_id":config["run_id"],"pid":os.getpid(),
                   "drive_run_folder_id":remote,**extra})
        print(f"pipeline phase={phase}",flush=True)
    def persist(path,relative):
        stat=signature(path)
        if previous.get(str(relative))==stat:return
        snapshot=make_stable_snapshot(path,temporary)
        if snapshot is None:return
        copy,original=snapshot
        parent_rel=relative.parent
        folder=remote
        sofar=Path()
        for part in parent_rel.parts:
            sofar=sofar/part; key=str(sofar)
            if key not in remote_dirs:remote_dirs[key]=find_or_create_folder(service,folder,part)
            folder=remote_dirs[key]
        file_id=upload_replace(service,folder,copy,remote_name=relative.name)
        received=temporary/"readback.bin"
        with received.open("wb") as stream:
            transfer=MediaIoBaseDownload(stream,service.files().get_media(fileId=file_id),chunksize=8*1024**2)
            done=False
            while not done:_,done=transfer.next_chunk(num_retries=2)
        if not filecmp.cmp(copy,received,shallow=False):raise IOError("Drive persistence byte readback differs")
        if signature(path)==original:previous[str(relative)]=original
        return {"relative":str(relative),"bytes":original[1],"file_id":file_id}
    def sync():
        with lock:
            verified=[]
            for directory,prefix in ((control,Path("control")),(output,Path())):
                if not directory.exists():continue
                for path in sorted(directory.rglob("*")):
                    if not path.is_file() or path.is_symlink() or path.suffix in (".tmp",".part"):continue
                    result=persist(path,prefix/path.relative_to(directory))
                    if result:verified.append(result)
            write_json(control/"persistence-state.json",{"utc":datetime.now(timezone.utc).isoformat(),
                        "state":"verified_readback","verified_files_this_pass":verified,
                        "total_currently_persisted_files":len(previous),"remote_run_folder_id":remote})
    def loop():
        while not stopped.wait(45):
            try:sync()
            except Exception as e:
                write_json(control/"persistence-error.json",{"utc":datetime.now(timezone.utc).isoformat(),"error":str(e)})
                print(f"Persistence error: {type(e).__name__}: {e}",flush=True)
    write_json(control/"pipeline-config.json",config)
    publish_state("persistence_probe")
    sync()  # A completed byte readback is required before restoration starts.
    threading.Thread(target=loop,daemon=True).start()
    def run(script):
        subprocess.run([sys.executable,str(root/script),"--config",str(args.config)],check=True)
    try:
        publish_state("restore_existing_features")
        run("restore_qc200.py")
        publish_state("source_cache_comparison")
        reference=Path(config["reference_features"])
        rows=json.loads(Path(config["reference_rows"]).read_text())
        results=[]
        with np.load(reference) as arrays:
            for row in rows:
                with h5py.File(Path(config["preproc_dir"])/row["relative"],"r") as file:cached=file["clip"][()]
                expected=arrays[row["key"]]
                equal=cached.shape==expected.shape and np.isfinite(cached).all() and np.allclose(cached,expected,rtol=1e-5,atol=1e-4)
                results.append({"reference_key":row["key"],"matches":bool(equal),
                                "max_abs_error":float(np.max(np.abs(cached-expected))) if cached.shape==expected.shape else None})
                if not equal:raise ValueError("Historical cache does not match source reference")
        evidence=control/"source-cache-agreement.json"
        write_json(evidence,{"scope":"representative training clips only, not a full raw-to-cache audit",
                   "recordings":len({r["relative"].split('.edf_')[0] for r in rows}),
                   "rtol":1e-5,"atol":1e-4,"results":results})
        provenance=control/"legacy-provenance.json"
        write_json(provenance,{"feature_version":config["feature_version"],"approved":True,
                   "reviewed_by":"root controlled execution: archived source and source-reference comparison",
                   "source_cache_agreement_verified":True,"scope":"sampled equivalence; full cached finite/shape audit follows",
                   "evidence_files":[str(evidence),str(reference),str(config["reference_rows"])]})
        config.update(feature_contract_approved=True,legacy_provenance_record=str(provenance),preprocess_mode="audit_cache")
        config["pause_file"]=str(control/"pause-training.request")
        write_json(args.config,config)
        write_json(control/"pipeline-config.json",config)
        publish_state("full_feature_audit")
        run("preprocess_qc200.py")
        sync()
        publish_state("fresh_normalization_then_B1_training")
        training=subprocess.Popen([sys.executable,str(root/"train_qc200.py"),"--config",str(args.config)])
        while training.poll() is None:
            state=json.loads((control/"persistence-state.json").read_text())
            verified_at=datetime.fromisoformat(state["utc"])
            if (datetime.now(timezone.utc)-verified_at).total_seconds()>300:
                Path(config["pause_file"]).write_text("Persistence has not completed a verified pass for 300 seconds")
            time.sleep(5)
        if training.returncode:raise RuntimeError(f"Training exited {training.returncode}")
        train_status=json.loads((output/"run-status.json").read_text())
        publish_state("B1_finished_train_dev_only" if train_status["state"]=="complete_train_dev" else "B1_paused",
                      training_state=train_status["state"])
        sync()
    except BaseException as error:
        publish_state("failed",error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
        try:sync()
        except Exception:traceback.print_exc()
        raise
    finally:
        stopped.set()

if __name__=="__main__":main()
