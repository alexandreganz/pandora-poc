"""
Cloudinary Uploader for Pandora Virtual Try-On

Uploads generated session images to Cloudinary for persistent storage.
Provides permanent URLs to view images.
"""

import os
import cloudinary
import cloudinary.uploader
from pathlib import Path


class CloudinaryUploader:
    """
    Handles uploading files to Cloudinary.
    """

    def __init__(self):
        """Initialize Cloudinary with environment variables."""
        self.enabled = False

        cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME")
        api_key = os.getenv("CLOUDINARY_API_KEY")
        api_secret = os.getenv("CLOUDINARY_API_SECRET")

        if all([cloud_name, api_key, api_secret]):
            cloudinary.config(
                cloud_name=cloud_name,
                api_key=api_key,
                api_secret=api_secret,
                secure=True
            )
            self.enabled = True
            print(f"[Cloudinary] Initialized successfully (cloud: {cloud_name})")
        else:
            print("[Cloudinary] Missing credentials - upload disabled")
            print("[Cloudinary] Set CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET")

    def upload_bytes(self, image_bytes: bytes, folder: str, filename: str) -> dict:
        """
        Upload image bytes to Cloudinary.

        Args:
            image_bytes: Raw image bytes
            folder: Folder path in Cloudinary (e.g., "pandora/session_123/poses")
            filename: Name for the file (without extension)

        Returns:
            Dict with 'success', 'url', 'public_id' or 'error'
        """
        if not self.enabled:
            return {"success": False, "error": "Cloudinary not configured"}

        try:
            result = cloudinary.uploader.upload(
                image_bytes,
                folder=folder,
                public_id=filename,
                resource_type="image",
                overwrite=True
            )

            url = result.get("secure_url")
            public_id = result.get("public_id")

            print(f"   [Cloudinary] Uploaded: {filename} -> {url}")

            return {
                "success": True,
                "url": url,
                "public_id": public_id
            }

        except Exception as e:
            print(f"   [Cloudinary] Upload failed for {filename}: {type(e).__name__}: {e}")
            return {"success": False, "error": str(e)}

    def upload_file(self, file_path: Path, folder: str) -> dict:
        """
        Upload a file from disk to Cloudinary.

        Args:
            file_path: Path to the file
            folder: Folder path in Cloudinary

        Returns:
            Dict with 'success', 'url', 'public_id' or 'error'
        """
        if not self.enabled:
            return {"success": False, "error": "Cloudinary not configured"}

        try:
            with open(file_path, "rb") as f:
                image_bytes = f.read()

            filename = file_path.stem  # filename without extension
            return self.upload_bytes(image_bytes, folder, filename)

        except Exception as e:
            print(f"   [Cloudinary] Upload failed for {file_path}: {type(e).__name__}: {e}")
            return {"success": False, "error": str(e)}


def upload_session_to_cloudinary(uploader: CloudinaryUploader, session_id: str, session_dir: Path) -> dict:
    """
    Upload an entire session folder to Cloudinary.

    Args:
        uploader: CloudinaryUploader instance
        session_id: Session identifier
        session_dir: Local path to session directory

    Returns:
        Dict with uploaded URLs organized by category
    """
    if not uploader or not uploader.enabled:
        print(f"[Cloudinary] Skipping upload - not enabled")
        return {}

    print(f"\n[Cloudinary] Starting upload for session: {session_id}")

    base_folder = f"pandora/{session_id}"
    urls = {
        "original": None,
        "poses": [],
        "try_ons": []
    }

    try:
        # Upload original photo
        original_photo = session_dir / "original_photo.jpg"
        if original_photo.exists():
            result = uploader.upload_file(original_photo, base_folder)
            if result["success"]:
                urls["original"] = result["url"]

        # Upload poses
        poses_dir = session_dir / "poses"
        if poses_dir.exists():
            for pose_file in sorted(poses_dir.glob("*.png")):
                result = uploader.upload_file(pose_file, f"{base_folder}/poses")
                if result["success"]:
                    urls["poses"].append({
                        "name": pose_file.stem,
                        "url": result["url"]
                    })

        # Upload try-ons
        tryons_dir = session_dir / "try_ons"
        if tryons_dir.exists():
            for tryon_file in sorted(tryons_dir.glob("*.png")):
                result = uploader.upload_file(tryon_file, f"{base_folder}/try_ons")
                if result["success"]:
                    urls["try_ons"].append({
                        "name": tryon_file.stem,
                        "url": result["url"]
                    })

        print(f"[Cloudinary] Session upload complete: {len(urls['poses'])} poses, {len(urls['try_ons'])} try-ons")
        return urls

    except Exception as e:
        print(f"[Cloudinary] Session upload failed: {type(e).__name__}: {e}")
        return urls
