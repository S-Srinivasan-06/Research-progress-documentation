"""Restore existing archives through authenticated Drive access; no reprocessing."""
import argparse
import json
from pathlib import Path
import shutil
from run_contract import load_config

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True,type=Path)
    args=parser.parse_args(); config=load_config(args.config)
    import google.auth
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    import stage_archive
    credentials,_=google.auth.default()
    service=build("drive","v3",credentials=credentials,cache_discovery=False)
    source_folder=json.loads(Path(config["restore_manifest"]).read_text())["source_folder_id"]
    def download(item,target):
        target=Path(target); target.parent.mkdir(parents=True,exist_ok=True)
        if target.is_file() and target.stat().st_size==item["size"]: return target
        metadata=service.files().get(fileId=item["id"],fields="id,name,size,trashed,mimeType,parents").execute()
        if metadata.get("trashed") or metadata["name"]!=item["name"] or int(metadata["size"])!=item["size"] or source_folder not in metadata.get("parents",[]):
            raise ValueError("Drive archive identity/size differs from saved manifest")
        if shutil.disk_usage(target.parent).free<item["size"]+2*1024**3:
            raise OSError("Insufficient archive download headroom")
        partial=target.with_suffix(target.suffix+".part")
        with partial.open("wb") as stream:
            downloader=MediaIoBaseDownload(stream,service.files().get_media(fileId=item["id"]),chunksize=16*1024**2)
            done=False; previous=-1
            while not done:
                progress,done=downloader.next_chunk(num_retries=2)
                bucket=int(progress.progress()*10) if progress else 0
                if bucket!=previous:
                    print(f"download {item['name']} {bucket*10}%",flush=True); previous=bucket
        if partial.stat().st_size!=item["size"]: raise IOError("Downloaded archive size mismatch")
        partial.replace(target); return target
    stage_archive.download=download
    # The original marker shards remain an independent coverage/provenance record.
    # Training uses the separately supplied QC-filtered marker directory.
    result=stage_archive.run(Path(config["restore_manifest"]),Path(config["restore_root"]),
                             None,False,Path(config["restore_status_dir"]),None)
    print(json.dumps(result,indent=2),flush=True)

if __name__=="__main__": main()
