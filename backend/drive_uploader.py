"""
Google Drive Uploader for Pandora Virtual Try-On

Uploads generated session images to Google Drive for backup/reference.
Uses service account authentication for headless operation.
"""

import os
import io
from pathlib import Path
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

# Scopes required for uploading files
SCOPES = ['https://www.googleapis.com/auth/drive.file']


class GoogleDriveUploader:
    """
    Handles uploading files and folders to Google Drive.

    Uses service account authentication - no user interaction required.
    """

    def __init__(self, service_account_file: str, root_folder_id: str):
        """
        Initialize the Drive uploader.

        Args:
            service_account_file: Path to service account JSON key file
            root_folder_id: Google Drive folder ID for uploads
        """
        self.root_folder_id = root_folder_id
        self.service = None
        self.enabled = False

        try:
            if os.path.exists(service_account_file):
                credentials = service_account.Credentials.from_service_account_file(
                    service_account_file, scopes=SCOPES
                )
                self.service = build('drive', 'v3', credentials=credentials)
                self.enabled = True
                print(f"[Drive] Initialized successfully")
            else:
                print(f"[Drive] Service account file not found: {service_account_file}")
                print(f"[Drive] Upload disabled - files will only be saved locally")
        except Exception as e:
            print(f"[Drive] Initialization failed: {type(e).__name__}: {e}")
            print(f"[Drive] Upload disabled - files will only be saved locally")

    def create_folder(self, folder_name: str, parent_id: str = None) -> str:
        """
        Create a folder in Google Drive.

        Args:
            folder_name: Name of the folder to create
            parent_id: Parent folder ID (uses root if None)

        Returns:
            Folder ID of the created folder, or None if failed
        """
        if not self.enabled:
            return None

        try:
            parent = parent_id or self.root_folder_id
            file_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': [parent]
            }
            folder = self.service.files().create(
                body=file_metadata,
                fields='id'
            ).execute()

            folder_id = folder.get('id')
            print(f"   [Drive] Created folder: {folder_name} (ID: {folder_id})")
            return folder_id

        except Exception as e:
            print(f"   [Drive] Failed to create folder {folder_name}: {type(e).__name__}: {e}")
            return None

    def upload_file(self, file_path: Path, parent_folder_id: str = None) -> str:
        """
        Upload a single file to Google Drive.

        Args:
            file_path: Local path to the file
            parent_folder_id: Parent folder ID (uses root if None)

        Returns:
            File ID of the uploaded file, or None if failed
        """
        if not self.enabled:
            return None

        try:
            parent = parent_folder_id or self.root_folder_id

            # Determine MIME type
            mime_type = 'image/png' if file_path.suffix.lower() == '.png' else 'image/jpeg'

            file_metadata = {
                'name': file_path.name,
                'parents': [parent]
            }

            with open(file_path, 'rb') as f:
                media = MediaIoBaseUpload(
                    io.BytesIO(f.read()),
                    mimetype=mime_type,
                    resumable=True
                )

                file = self.service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields='id'
                ).execute()

            file_id = file.get('id')
            print(f"   [Drive] Uploaded: {file_path.name} (ID: {file_id})")
            return file_id

        except Exception as e:
            print(f"   [Drive] Failed to upload {file_path.name}: {type(e).__name__}: {e}")
            return None


def upload_session_to_drive(uploader: GoogleDriveUploader, session_id: str, session_dir: Path):
    """
    Upload an entire session folder to Google Drive.

    Creates the same folder structure in Drive as exists locally:
    - session_id/
      - original_photo.jpg
      - poses/
        - pose_*.png
      - try_ons/
        - *.png

    Args:
        uploader: GoogleDriveUploader instance
        session_id: Session identifier (used as folder name)
        session_dir: Local path to session directory
    """
    if not uploader or not uploader.enabled:
        print(f"[Drive] Skipping upload - Drive not enabled")
        return

    print(f"\n[Drive] Starting background upload for session: {session_id}")

    try:
        # Create session folder in Drive
        session_folder_id = uploader.create_folder(session_id)
        if not session_folder_id:
            print(f"[Drive] Failed to create session folder, aborting upload")
            return

        # Upload original photo if exists
        original_photo = session_dir / "original_photo.jpg"
        if original_photo.exists():
            uploader.upload_file(original_photo, session_folder_id)

        # Upload poses folder
        poses_dir = session_dir / "poses"
        if poses_dir.exists() and any(poses_dir.iterdir()):
            poses_folder_id = uploader.create_folder("poses", session_folder_id)
            if poses_folder_id:
                for pose_file in poses_dir.glob("*.png"):
                    uploader.upload_file(pose_file, poses_folder_id)

        # Upload try_ons folder
        tryons_dir = session_dir / "try_ons"
        if tryons_dir.exists() and any(tryons_dir.iterdir()):
            tryons_folder_id = uploader.create_folder("try_ons", session_folder_id)
            if tryons_folder_id:
                for tryon_file in tryons_dir.glob("*.png"):
                    uploader.upload_file(tryon_file, tryons_folder_id)

        print(f"[Drive] Session upload complete: {session_id}")

    except Exception as e:
        print(f"[Drive] Session upload failed: {type(e).__name__}: {e}")
