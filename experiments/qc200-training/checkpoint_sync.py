"""Copy a stopped/paused run to an existing mounted persistence destination."""
import argparse
from pathlib import Path
import shutil
from datetime import datetime, timezone
from run_contract import load_config, write_json

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True,type=Path)
    args=parser.parse_args(); config=load_config(args.config)
    source=Path(config["output_dir"]).resolve()
    parent=Path(config["checkpoint_destination"]).resolve()
    if not parent.is_dir(): raise ValueError("Persistence destination must already exist/mount")
    target=parent/config["run_id"]
    if source==target or source in target.parents or target in source.parents:
        raise ValueError("Source/destination overlap; mounted-direct outputs already persist")
    status_path=source/"run-status.json"
    if not status_path.exists(): raise ValueError("Sync only a paused/completed run")
    import json
    if json.loads(status_path.read_text())["state"] not in ("paused_runtime","complete_train_dev"):
        raise ValueError("Stop/pause the trainer before snapshot synchronization")
    identity=json.loads((source/"run_identity.json").read_text())
    if identity["run_id"]!=config["run_id"]: raise ValueError("Source run identity differs")
    if identity["variant"]!=config.get("variant","B1") or identity["feature_version"]!=config["feature_version"]:
        raise ValueError("Sync configuration differs from saved experiment")
    if target.exists():
        existing=target/"run_identity.json"
        if not existing.is_file() or json.loads(existing.read_text())!=identity:
            raise ValueError("Persistence destination contains a different or unidentifiable run")
    copied=[]
    paths=sorted(source.rglob("*"),key=lambda p:(p.name in ("run-status.json","drive-sync-state.json"),str(p)))
    for path in paths:
        if not path.is_file() or path.suffix==".tmp": continue
        relative=path.relative_to(source); dest=target/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        temp=dest.with_name(dest.name+".uploading")
        shutil.copyfile(path,temp)
        # Bounded-memory direct byte readback, no hashing.
        with path.open("rb") as a,temp.open("rb") as b:
            while True:
                block=a.read(1024*1024); other=b.read(1024*1024)
                if block!=other: raise IOError("Persistence readback differs")
                if not block: break
        temp.replace(dest); copied.append({"name":str(relative),"bytes":dest.stat().st_size})
    state={"utc":datetime.now(timezone.utc).isoformat(),"state":"verified_readback","files":copied}
    write_json(source/"drive-sync-state.json",state)
    write_json(target/"drive-sync-state.json",state)
    print(f"Verified {len(copied)} persisted files")

if __name__=="__main__": main()
