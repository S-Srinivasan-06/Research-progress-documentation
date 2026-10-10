"""Drive API helpers reused from the archived synchronizer; no historical resume IDs."""

from pathlib import Path

import time, shutil

RETRIES = 3

def drive_client():
    import google.auth
    from googleapiclient.discovery import build

    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/drive"])
    return build("drive", "v3", credentials=creds, cache_discovery=False)

def escape_query(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")

def list_children(service, parent_id: str, name: str | None = None) -> list[dict]:
    clauses = [f"'{escape_query(parent_id)}' in parents", "trashed = false"]
    if name is not None:
        clauses.append(f"name = '{escape_query(name)}'")
    response = service.files().list(
        q=" and ".join(clauses),
        pageSize=1000,
        fields="nextPageToken,files(id,name,mimeType,size,modifiedTime)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    files = list(response.get("files", []))
    while response.get("nextPageToken"):
        response = service.files().list(
            q=" and ".join(clauses), pageSize=1000,
            pageToken=response["nextPageToken"],
            fields="nextPageToken,files(id,name,mimeType,size,modifiedTime)",
            supportsAllDrives=True, includeItemsFromAllDrives=True,
        ).execute()
        files.extend(response.get("files", []))
    return files

def find_or_create_folder(service, parent_id: str, name: str) -> str:
    matches = [item for item in list_children(service, parent_id, name)
               if item.get("mimeType") == "application/vnd.google-apps.folder"]
    if len(matches) > 1:
        raise RuntimeError(f"Multiple Drive folders named {name!r} under the selected parent.")
    if matches:
        return matches[0]["id"]
    body = {"name": name, "mimeType": "application/vnd.google-apps.folder", "parents": [parent_id]}
    result = service.files().create(body=body, fields="id,name", supportsAllDrives=True).execute()
    return result["id"]

def upload_replace(service, folder_id: str, local_path: Path, remote_name: str | None = None) -> str:
    from googleapiclient.http import MediaFileUpload
    remote_name = remote_name or local_path.name
    delay = 2
    for attempt in range(RETRIES):
        media = MediaFileUpload(str(local_path), mimetype="application/octet-stream",
                                resumable=True, chunksize=8 * 1024 * 1024)
        try:
            # Requery on every retry: a timed-out create may have committed remotely.
            matches = list_children(service, folder_id, remote_name)
            files = [x for x in matches if x.get("mimeType") != "application/vnd.google-apps.folder"]
            if len(files) > 1:
                raise RuntimeError(f"Duplicate Drive output files named {remote_name!r}.")
            if files:
                request = service.files().update(
                    fileId=files[0]["id"], media_body=media,
                    fields="id,name,size,modifiedTime", supportsAllDrives=True,
                )
            else:
                request = service.files().create(
                    body={"name": remote_name, "parents": [folder_id]},
                    media_body=media, fields="id,name,size,modifiedTime", supportsAllDrives=True,
                )
            response = None
            while response is None:
                _, response = request.next_chunk()
            return response["id"]
        except Exception:
            if attempt + 1 == RETRIES:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 60)
    raise AssertionError("unreachable")

def signature(path: Path) -> tuple[int, int, int]:
    st = path.stat()
    return st.st_ino, st.st_size, st.st_mtime_ns

def make_stable_snapshot(source: Path, temp_dir: Path) -> tuple[Path, tuple[int, int, int]] | None:
    before = signature(source)
    target = temp_dir / (source.name + ".snapshot")
    shutil.copy2(source, target)
    after = signature(source)
    if before != after or target.stat().st_size != before[1]:
        target.unlink(missing_ok=True)
        return None
    return target, before
