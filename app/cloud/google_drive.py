import datetime
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List

from app.database import DATA_DIR

logger = logging.getLogger("GoogleDriveSync")
DEFAULT_SCOPES = ["https://www.googleapis.com/auth/drive.file"]


class GoogleDriveManager:
    CREDENTIALS_FILE: Path = DATA_DIR / "credentials.json"
    TOKEN_FILE: Path = DATA_DIR / "token.json"

    def __init__(self):
        self.service = None

    def authenticate(self) -> None:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build

        if not self.CREDENTIALS_FILE.exists():
            raise FileNotFoundError(f"Brak pliku {self.CREDENTIALS_FILE}")

        creds = None
        if self.TOKEN_FILE.exists():
            # Do not override scopes if token.json already exists
            creds = Credentials.from_authorized_user_file(str(self.TOKEN_FILE))

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(str(self.CREDENTIALS_FILE), DEFAULT_SCOPES)
                creds = flow.run_local_server(port=0, open_browser=True)
            self.TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")

        self.service = build("drive", "v3", credentials=creds)

    @staticmethod
    def compute_local_md5(file_path: Path) -> str:
        h = hashlib.md5()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    def list_folders(self) -> List[Dict[str, str]]:
        if not self.service:
            self.authenticate()
        q = "mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        res = self.service.files().list(
            q=q, spaces="drive", fields="files(id, name)", orderBy="name", pageSize=100
        ).execute()
        return res.get("files", [])

    def get_or_create_folder(self, folder_name: str) -> str:
        if not self.service:
            self.authenticate()
        clean_name = folder_name.strip()
        q = f"name = '{clean_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        res = self.service.files().list(q=q, spaces="drive", fields="files(id, name)").execute()
        files = res.get("files", [])
        if files:
            return files[0]["id"]

        meta = {"name": clean_name, "mimeType": "application/vnd.google-apps.folder"}
        created = self.service.files().create(body=meta, fields="id").execute()
        return created["id"]

    def list_files_in_folder(self, folder_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        if not self.service:
            self.authenticate()
        q = f"'{folder_id}' in parents and trashed = false and mimeType != 'application/vnd.google-apps.folder'"
        res = self.service.files().list(
            q=q,
            spaces="drive",
            fields="files(id, name, size, md5Checksum, createdTime, webViewLink)",
            orderBy="createdTime desc",
            pageSize=limit,
        ).execute()
        return res.get("files", [])

    def upload_and_verify(
        self, local_path: Path, folder_id: str, purge_local_after_verify: bool = False
    ) -> Dict[str, Any]:
        if not self.service:
            self.authenticate()
        if not local_path.exists():
            raise FileNotFoundError(f"Plik lokalny nie istnieje: {local_path}")

        from googleapiclient.http import MediaFileUpload

        local_size = local_path.stat().st_size
        local_md5 = self.compute_local_md5(local_path)

        meta = {"name": local_path.name, "parents": [folder_id]}
        media = MediaFileUpload(str(local_path.resolve()), chunksize=5 * 1024 * 1024, resumable=True)
        req = self.service.files().create(
            body=meta,
            media_body=media,
            fields="id, name, size, md5Checksum, createdTime, webViewLink",
        )

        response = None
        while response is None:
            _, response = req.next_chunk()

        cloud_size = int(response.get("size", -1))
        cloud_md5 = response.get("md5Checksum", "")

        verified = (cloud_size == local_size) and (cloud_md5.lower() == local_md5.lower())
        if not verified:
            raise RuntimeError(
                f"Niezgodność sumy kontrolnej dla {local_path.name}! "
                f"Lokalny MD5={local_md5} vs Cloud MD5={cloud_md5}. Plik lokalny NIE został usunięty."
            )

        purged = False
        if purge_local_after_verify and verified:
            local_path.unlink()
            purged = True

        return {
            "id": response.get("id"),
            "name": response.get("name"),
            "size_bytes": cloud_size,
            "md5": cloud_md5,
            "verified": verified,
            "purged_local": purged,
            "webViewLink": response.get("webViewLink"),
            "finished_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }