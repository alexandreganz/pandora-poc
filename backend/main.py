import os
import sys
import base64
import random
import time
import uuid
import io
import asyncio
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from google import genai
from google.genai import types
from drive_uploader import GoogleDriveUploader, upload_session_to_drive
from cloudinary_uploader import CloudinaryUploader, upload_session_to_cloudinary

# Fix Windows console encoding for Unicode characters (✓, ❌, etc.)
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        # Python < 3.7 fallback
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

load_dotenv()

app = FastAPI(title="Pandora API", version="1.0.0")

# CORS for frontend (supports environment-based origins for production)
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:5174")
allowed_origins = [origin.strip() for origin in cors_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini with new SDK
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_ID = 'gemini-3-pro-image-preview'  # Primary model for image generation

# Standard iris diameter constant for PPM calculation
HVID_MM = 11.7

# Path to earring assets (transparent PNGs)
ASSETS_PATH = Path(__file__).parent.parent / "frontend" / "public" / "images"

# Output directories for generated images
OUTPUT_BASE_PATH = Path(__file__).parent / "generated_images"
POSES_OUTPUT_PATH = OUTPUT_BASE_PATH / "poses"
TRYON_OUTPUT_PATH = OUTPUT_BASE_PATH / "try_ons"

# Create output directories if they don't exist
for path in [OUTPUT_BASE_PATH, POSES_OUTPUT_PATH, TRYON_OUTPUT_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# Initialize Google Drive uploader (optional - for backup)
DRIVE_ENABLED = os.getenv("GOOGLE_DRIVE_UPLOAD_ENABLED", "false").lower() == "true"
if DRIVE_ENABLED:
    drive_uploader = GoogleDriveUploader(
        service_account_file=os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "service-account.json"),
        root_folder_id=os.getenv("GOOGLE_DRIVE_FOLDER_ID", "")
    )
else:
    drive_uploader = None
    print("[Drive] Upload disabled via environment variable")

# Initialize Cloudinary uploader (optional - for cloud storage)
CLOUDINARY_ENABLED = os.getenv("CLOUDINARY_CLOUD_NAME") is not None
if CLOUDINARY_ENABLED:
    cloudinary_uploader = CloudinaryUploader()
else:
    cloudinary_uploader = None
    print("[Cloudinary] Upload disabled - no credentials configured")

# Product data with dimensions in mm for proper scaling
PRODUCTS = [
    {
        "id": 1,
        "name": "Classic Circle",
        "price": 89.00,
        "image1": "/images/classic-circle-1.jpg",
        "image2": "/images/classic-circle-2.jpg",
        "asset": "classic-circle-1.jpg",
        "height_mm": 25.0,
        "width_mm": 25.0
    },
    {
        "id": 2,
        "name": "Silver Flower",
        "price": 95.00,
        "image1": "/images/silver-flower-1.jpg",
        "image2": "/images/silver-flower-2.jpg",
        "asset": "silver-flower-1.jpg",
        "height_mm": 6.7,
        "width_mm": 6.8
    },
    {
        "id": 3,
        "name": "Red Heart",
        "price": 79.00,
        "image1": "/images/red-heart-1.jpg",
        "image2": "/images/red-heart-2.jpg",
        "asset": "red-heart-1.jpg",
        "height_mm": 15.0,
        "width_mm": 14.0
    },
    {
        "id": 4,
        "name": "Gold Heart",
        "price": 125.00,
        "image1": "/images/gold-heart-1.jpg",
        "image2": "/images/gold-heart-2.jpg",
        "asset": "gold-heart-1.jpg",
        "height_mm": 18.0,
        "width_mm": 16.0
    },
    {
        "id": 5,
        "name": "Silver Heart",
        "price": 85.00,
        "image1": "/images/silver-heart-1.jpg",
        "image2": "/images/silver-heart-2.jpg",
        "asset": "silver-heart-1.jpg",
        "height_mm": 15.0,
        "width_mm": 14.0
    },
    {
        "id": 6,
        "name": "Blue Butterfly",
        "price": 110.00,
        "image1": "/images/blue-butterfly-1.jpg",
        "image2": "/images/blue-butterfly-2.jpg",
        "asset": "blue-butterfly-1.jpg",
        "height_mm": 20.0,
        "width_mm": 22.0
    }
]


class PhotoScaler:
    """
    Iris-based photo scaling for accurate earring sizing.

    Uses MediaPipe face mesh to detect iris diameter, then calculates
    pixels-per-millimeter (PPM) ratio to scale earring images correctly.
    """

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.hvid_mm = HVID_MM  # Standard horizontal visible iris diameter

    def calculate_ppm(self, image_bytes: bytes) -> float:
        """
        Calculate pixels-per-millimeter from iris detection.

        Args:
            image_bytes: Raw image bytes (JPEG/PNG)

        Returns:
            PPM value (pixels per millimeter), or None if detection fails
        """
        try:
            # Convert bytes to numpy array for OpenCV
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                print("   [ERROR] Failed to decode image for iris detection")
                return None

            h, w, _ = image.shape
            print(f"   → Image dimensions: {w}x{h} pixels")

            with self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            ) as face_mesh:

                results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                if not results.multi_face_landmarks:
                    print("   [ERROR] Face detection failed - no face landmarks found")
                    return None

                # Get iris landmarks (indices 474 and 476 for right iris)
                mesh_points = results.multi_face_landmarks[0].landmark
                p1 = mesh_points[474]  # Left side of right iris
                p2 = mesh_points[476]  # Right side of right iris

                # Calculate horizontal pixel distance
                dist_px = (((p2.x - p1.x) * w)**2 + ((p2.y - p1.y) * h)**2)**0.5
                ppm = dist_px / self.hvid_mm

                print(f"   → Iris width: {dist_px:.2f} pixels")
                print(f"   → Scale factor: {ppm:.4f} pixels/mm")

                return ppm

        except Exception as e:
            print(f"   [ERROR] Iris detection failed: {type(e).__name__}: {str(e)}")
            return None

    def scale_earring(self, earring_bytes: bytes, height_mm: float, width_mm: float, ppm: float) -> bytes:
        """
        Scale earring image to correct size based on PPM.

        Args:
            earring_bytes: Raw earring image bytes (PNG with transparency)
            height_mm: Earring height in millimeters
            width_mm: Earring width in millimeters
            ppm: Pixels per millimeter from iris detection

        Returns:
            Scaled earring image bytes (PNG), or original if scaling fails
        """
        try:
            # Load earring image
            img = Image.open(io.BytesIO(earring_bytes))

            # Calculate target dimensions in pixels
            target_h_px = int(height_mm * ppm)
            target_w_px = int(width_mm * ppm)

            print(f"   → Original earring size: {img.width}x{img.height} pixels")
            print(f"   → Target size: {target_w_px}x{target_h_px} pixels (from {width_mm}mm x {height_mm}mm)")

            # Resize using LANCZOS for best quality
            resized = img.resize((target_w_px, target_h_px), Image.Resampling.LANCZOS)

            # Convert back to bytes
            output = io.BytesIO()
            resized.save(output, format='PNG')
            output.seek(0)

            print(f"   [OK] Earring scaled successfully")
            return output.read()

        except Exception as e:
            print(f"   [ERROR] Earring scaling failed: {type(e).__name__}: {str(e)}")
            return earring_bytes  # Return original if scaling fails


# Initialize photo scaler
photo_scaler = PhotoScaler()


# Target width for optimized images (balances quality vs upload speed)
OPTIMIZED_WIDTH = 1280


def optimize_image(image_bytes: bytes, target_width: int = OPTIMIZED_WIDTH, quality: int = 85) -> tuple[bytes, float]:
    """
    Resize image to target width to reduce upload size for API calls.

    Args:
        image_bytes: Original image bytes
        target_width: Target width in pixels (height scales proportionally)
        quality: JPEG quality (1-100)

    Returns:
        Tuple of (optimized_bytes, scale_ratio)
        scale_ratio is new_width/original_width for PPM adjustment
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        original_width, original_height = img.size

        # Skip if already smaller than target
        if original_width <= target_width:
            print(f"   [OK] Image already optimized ({original_width}x{original_height})")
            return image_bytes, 1.0

        # Calculate new dimensions maintaining aspect ratio
        scale_ratio = target_width / original_width
        new_height = int(original_height * scale_ratio)

        # Resize with high-quality resampling
        resized = img.resize((target_width, new_height), Image.Resampling.LANCZOS)

        # Convert to RGB if necessary (for JPEG output)
        if resized.mode in ('RGBA', 'P'):
            resized = resized.convert('RGB')

        # Save as JPEG with specified quality
        output = io.BytesIO()
        resized.save(output, format='JPEG', quality=quality, optimize=True)
        output.seek(0)
        optimized_bytes = output.read()

        original_kb = len(image_bytes) / 1024
        optimized_kb = len(optimized_bytes) / 1024
        reduction = (1 - optimized_kb / original_kb) * 100

        print(f"   [OK] Image optimized: {original_width}x{original_height} -> {target_width}x{new_height}")
        print(f"   [OK] Size reduced: {original_kb:.0f}KB -> {optimized_kb:.0f}KB ({reduction:.0f}% reduction)")

        return optimized_bytes, scale_ratio

    except Exception as e:
        print(f"   [WARNING] Image optimization failed: {type(e).__name__}: {e}")
        return image_bytes, 1.0


class VirtualTryOnGenerator:
    """
    Two-Stage AI Virtual Try-On Workflow

    Stage 1: Identity-Preserving Pose Variation
        - Takes front face photo + style reference
        - Generates pose variation while maintaining identity
        - Ensures ear visibility for jewelry placement

    Stage 2: Earring Composition
        - Takes posed image + earring asset + style reference
        - Composites earring onto the pose
        - Maintains lighting consistency
    """

    def __init__(self, client: genai.Client, model_id: str, poses_dir: Path = None, tryon_dir: Path = None):
        self.client = client
        self.model_id = model_id
        self.poses_dir = poses_dir or Path("poses")
        self.tryon_dir = tryon_dir or Path("try_ons")

        # Ensure directories exist
        self.poses_dir.mkdir(parents=True, exist_ok=True)
        self.tryon_dir.mkdir(parents=True, exist_ok=True)

        self.target_poses = {
            # --- EXISTING REFINED ---
            "profile_left": "a strict 90-degree side profile looking left, emphasizing the sharp jawline and clear earlobe silhouette",
            "three_quarter_right": "a classic three-quarter view looking slightly right, focusing on the ear and the soft curve of the cheek",
            "slightly_up": "a slight upward gaze and chin tilt, lengthening the neck and providing an unobstructed view of the earlobe",
            
            # --- NEW BRAND-ALIGNED POSES ---
            "editorial_tilt_down": "a subtle downward chin tuck while looking toward the camera, highlighting the top curve of the earring and the temple",
            "over_shoulder_glance": "an over-the-shoulder view with the head turned back toward the lens, showcasing the ear and the back of the jaw",
            "close_up_macro_ear": "an extreme macro side-view focusing exclusively on the ear and the area between the temple and the jawline",
            "soft_chin_rest": "a three-quarter profile with the head resting slightly toward one shoulder, creating a soft, elegant angle for the jewelry",
            "profile_zenith_gaze": "a side profile with the gaze directed far upward, pulling the skin taut across the jawline for a clean, architectural look",
            "direct_ear_parallel": "a side view where the camera is perfectly parallel to the ear, capturing the earring's design with zero perspective distortion",
            "low_angle_authority": "a slight low-angle view looking up toward the jaw and ear, giving the jewelry a more prominent, heroic presence"
        }

    def _save_image_from_response(self, response) -> bytes:
        """Extract image bytes from Gemini response."""
        try:
            if not response.candidates:
                print("   [ERROR] No candidates in response")
                return None

            candidate = response.candidates[0]
            if not candidate.content or not candidate.content.parts:
                print("   [ERROR] No content parts in response candidate")
                return None

            for part in candidate.content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    return part.inline_data.data

            return None
        except (AttributeError, IndexError) as e:
            print(f"   [ERROR] Image extraction error: {type(e).__name__}: {e}")
            return None
        except Exception as e:
            print(f"   [ERROR] Unexpected image extraction error: {type(e).__name__}: {e}")
            return None

    def generate_styled_pose(self, face_bytes: bytes, style_ref_bytes: bytes, pose_key: str = None, session_id: str = None) -> bytes:
        """
        Stage 1: Identity-Preserving Pose Variation

        Args:
            face_bytes: Original front-facing photo
            style_ref_bytes: Style reference for lighting/quality
            pose_key: Optional specific pose (defaults to random)
            session_id: Optional session ID for organizing saved files

        Returns:
            Image bytes of posed variation
        """
        if pose_key is None:
            pose_key = random.choice(list(self.target_poses.keys()))

        pose_desc = self.target_poses[pose_key]
        print(f"   → Pose type: {pose_key} - {pose_desc}")
        print(f"   → Face photo size: {len(face_bytes)} bytes")
        print(f"   → Style reference size: {len(style_ref_bytes)} bytes")

        content_parts = [
            types.Part.from_bytes(data=face_bytes, mime_type="image/jpeg"),
            types.Part.from_bytes(data=style_ref_bytes, mime_type="image/jpeg"),
            f"TASK: Generate a new pose of the EXACT person from Image 1. "
            f"IDENTITY (CRITICAL - 100% PRESERVATION): "
            f"The new background MUST match the clean, seamless studio aesthetic of Image 3. "
            f"- Face: Keep the EXACT same face shape, eyes, nose, mouth, eyebrows, and all facial features from Image 1. "
            f"- Skin: Keep the EXACT same skin tone, texture, and complexion from Image 1. "
            f"- Hair: Keep the EXACT same hair color, style, length, and texture from Image 1. "
            f"- Age: Keep the EXACT same apparent age from Image 1. "
            f"- Ethnicity: Keep the EXACT same ethnic features from Image 1. "
            f"DO NOT borrow ANY physical features from Image 2. The person must be 100% recognizable as the same individual from Image 1. "
            f"FROM IMAGE 2 USE ONLY: Lighting direction, lighting quality, shadow softness, and overall photographic style. "
            f"POSE: Re-imagine this exact person in {pose_desc}. "
            f"CRITICAL: Expose the ear clearly for jewelry placement. "
            f"OUTPUT: A photorealistic image that looks exactly like the person from Image 1, just in a different pose."
        ]

        try:
            print(f"   → Calling Gemini API ({self.model_id})...")
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=content_parts,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"]
                )
            )
            print(f"   [OK] Gemini API responded")

            pose_bytes = self._save_image_from_response(response)

            if not pose_bytes:
                # Log the response for debugging
                print(f"   [ERROR] Failed to extract image from Gemini response")
                if response.candidates and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'text') and part.text:
                            print(f"   [DEBUG] Text response: {part.text[:200]}...")
                return None

            print(f"   [OK] Extracted image: {len(pose_bytes)} bytes")

            # Save pose image to disk (in session-specific folder)
            if pose_bytes and session_id:
                session_poses_dir = OUTPUT_BASE_PATH / session_id / "poses"
                session_poses_dir.mkdir(parents=True, exist_ok=True)
                filename = f"pose_{pose_key}_{int(time.time())}.png"
                filepath = session_poses_dir / filename
                with open(filepath, "wb") as f:
                    f.write(pose_bytes)
                print(f"   [OK] Saved pose to: {filepath}")

            return pose_bytes
        except Exception as e:
            print(f"   [ERROR] STAGE 1 EXCEPTION: {type(e).__name__}: {str(e)}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")
            return None

    def generate_final_vto(self, pose_bytes: bytes, earring_bytes: bytes, style_bytes: bytes, session_id: str = None, product_name: str = None) -> bytes:
        """
        Stage 2: Earring Composition onto Pose

        Args:
            pose_bytes: Posed image from Stage 1
            earring_bytes: Earring asset (transparent PNG)
            style_bytes: Style reference for lighting
            session_id: Optional session ID for organizing saved files
            product_name: Optional product name for filename

        Returns:
            Final VTO image bytes
        """
        print(f"      → Pose size: {len(pose_bytes)} bytes")
        print(f"      → Earring size: {len(earring_bytes)} bytes")
        print(f"      → Style reference size: {len(style_bytes)} bytes")

        content_parts = [
            types.Part.from_bytes(data=pose_bytes, mime_type="image/png"),   # Reference 1: Person
            types.Part.from_bytes(data=earring_bytes, mime_type="image/png"),# Reference 2: Earring
            types.Part.from_bytes(data=style_bytes, mime_type="image/jpeg"), # Reference 3: Style
        # 1. TASK OVERVIEW
            f"TASK: Add the earring from Image 2 onto the person in Image 1. ",

            # 2. IDENTITY CONSTRAINTS (IMAGE 1)
            f"IDENTITY CONSTRAINT (CRITICAL): Keep 100% of the person's appearance from Image 1: face, skin tone, hair, and features. "
            f"Do NOT borrow any physical features from the person in Image 3. No alterations to the model's anatomy. "

            # 3. OBJECT CONSTRAINTS (IMAGE 2)
            f"OBJECT CONSTRAINT (CRITICAL): Use ONLY the exact {product_name} design from Image 2. "
            f"Maintain an EXACT PIXEL MATCH of its shape, size, and proportions. "
            f"Do NOT resize, rescale, shrink, enlarge, or warp the earring design under any circumstances. "

            # 4. PLACEMENT & STYLING
            f"PLACEMENT: Place exactly ONE earring on the visible earlobe. Do NOT add extra earrings or piercings. "
            f"WARDROBE: Maintain the minimalist styling of Image 1 (e.g., simple black/neutral tank or bare shoulders). "
            f"The new background MUST match the clean, seamless studio aesthetic of Image 3. "

            # 5. BRAND DNA & TEXTURE
            f"BRAND DNA: Maintain hyper-realistic skin texture. Do NOT smooth, airbrush, or apply 'beauty' filters. "
            f"Ensure natural pores, freckles, and fine lines are visible as seen in the brand's core imagery. "

            # 6. LIGHTING & BACKGROUND
            f"LIGHTING & STYLE: Apply the soft-box studio lighting and precise earring reflections from Image 3. "
            f"BACKGROUND: Use a clean, seamless, softly blurred studio white or light grey background. "
            f"Do NOT use dark, colored, or textured backgrounds. Ensure a shallow depth of field. "

            # 7. FINAL OUTPUT
            f"OUTPUT: A high-end, authentic macro editorial portrait. The exact person from Image 1 wearing the exact earring from Image 2."
        ]
         

        try:
            print(f"      → Calling Gemini API ({self.model_id})...")
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=content_parts,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"]
                )
            )
            print(f"      [OK] Gemini API responded")

            final_bytes = self._save_image_from_response(response)

            if not final_bytes:
                # Log the response for debugging
                print(f"      [ERROR] Failed to extract image from Gemini response")
                if response.candidates and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'text') and part.text:
                            print(f"      [DEBUG] Text response: {part.text[:200]}...")
                return None

            print(f"      [OK] Extracted final image: {len(final_bytes)} bytes")

            # Save final try-on image to disk (in session-specific folder)
            if final_bytes and session_id:
                session_tryons_dir = OUTPUT_BASE_PATH / session_id / "try_ons"
                session_tryons_dir.mkdir(parents=True, exist_ok=True)
                safe_product_name = product_name.replace(" ", "_").lower() if product_name else "unknown"
                filename = f"{safe_product_name}_{int(time.time())}.png"
                filepath = session_tryons_dir / filename
                with open(filepath, "wb") as f:
                    f.write(final_bytes)
                print(f"      [OK] Saved to disk: {filepath}")

            return final_bytes
        except Exception as e:
            print(f"      [ERROR] STAGE 2 EXCEPTION: {type(e).__name__}: {str(e)}")
            import traceback
            print(f"      Traceback: {traceback.format_exc()}")
            return None

    def run_workflow(self, face_bytes: bytes, earring_bytes: bytes, style_ref_bytes: bytes) -> bytes:
        """
        Execute full two-stage workflow.

        Returns:
            Final VTO image bytes
        """
        # Stage 1: Generate posed variation
        pose_bytes = self.generate_styled_pose(face_bytes, style_ref_bytes)

        if not pose_bytes:
            return None

        # Stage 2: Composite earring
        final_bytes = self.generate_final_vto(pose_bytes, earring_bytes, style_ref_bytes)

        return final_bytes


# Initialize generator with output directories
vto_generator = VirtualTryOnGenerator(
    gemini_client,
    MODEL_ID,
    poses_dir=POSES_OUTPUT_PATH,
    tryon_dir=TRYON_OUTPUT_PATH
)


@app.get("/api/products")
async def get_products():
    """Return all products."""
    return {"products": PRODUCTS}


@app.post("/api/try-on-all")
async def try_on_all(
    front_photo: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    AI Virtual Try-On: Generate 5 photos (one for each earring model)

    Two-Stage Workflow:
    1. Generate styled pose variation from front photo
    2. Composite each earring model onto the pose

    Returns: Array of 5 generated try-on images
    All generated images are saved to disk.
    """
    try:
        # Generate unique session ID with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"

        # Create session directory
        session_dir = OUTPUT_BASE_PATH / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"New Try-On Session: {session_id}")
        print(f"Output Directory: {session_dir}")
        print(f"{'='*60}\n")

        # Read front photo
        front_contents = await front_photo.read()

        # Save original uploaded photo
        original_photo_path = session_dir / "original_photo.jpg"
        with open(original_photo_path, "wb") as f:
            f.write(front_contents)
        print(f"[OK] Saved original photo: {original_photo_path}")

        # ============================================
        # IRIS DETECTION: Calculate PPM for scaling
        # ============================================
        print("\n" + "="*60)
        print("IRIS DETECTION: Calculating Scale Factor")
        print("="*60)
        ppm = photo_scaler.calculate_ppm(front_contents)
        if ppm:
            print(f"[OK] PPM calculated: {ppm:.4f} pixels/mm (from original resolution)")
        else:
            print("[WARNING] PPM calculation failed - using default scaling")
            ppm = None
        print("="*60 + "\n")

        # ============================================
        # OPTIMIZE IMAGE: Reduce upload size for faster API calls
        # ============================================
        print("="*60)
        print("IMAGE OPTIMIZATION: Reducing upload size")
        print("="*60)
        front_contents, scale_ratio = optimize_image(front_contents)

        # Adjust PPM for the resized image
        if ppm and scale_ratio != 1.0:
            ppm = ppm * scale_ratio
            print(f"[OK] PPM adjusted for resize: {ppm:.4f} pixels/mm")
        print("="*60 + "\n")

        # ============================================
        # PREPARE DATA FOR PARALLEL PROCESSING
        # ============================================
        pose_keys = list(vto_generator.target_poses.keys())
        random.shuffle(pose_keys)

        print("\n" + "="*60)
        print("PREPARING PRODUCTS FOR PARALLEL PROCESSING")
        print("="*60)

        # Semaphore to limit concurrent API calls (avoid rate limits)
        semaphore = asyncio.Semaphore(3)

        # Prepare all product data upfront
        product_data = []
        for idx, product in enumerate(PRODUCTS):
            # Load and optimize style reference
            style_ref_filename = product["image2"].replace("/images/", "")
            style_ref_path = ASSETS_PATH / style_ref_filename

            if style_ref_path.exists():
                with open(style_ref_path, "rb") as f:
                    style_bytes = f.read()
                style_bytes, _ = optimize_image(style_bytes)
            else:
                style_bytes = front_contents

            # Load and scale earring asset
            earring_asset_path = ASSETS_PATH / product["asset"]
            if not earring_asset_path.exists():
                print(f"   [ERROR] Earring asset not found: {product['asset']}")
                continue

            with open(earring_asset_path, "rb") as f:
                earring_bytes = f.read()

            if ppm and "height_mm" in product and "width_mm" in product:
                earring_bytes = photo_scaler.scale_earring(
                    earring_bytes=earring_bytes,
                    height_mm=product["height_mm"],
                    width_mm=product["width_mm"],
                    ppm=ppm
                )

            pose_key = pose_keys[idx % len(pose_keys)]

            product_data.append({
                "product": product,
                "style_bytes": style_bytes,
                "earring_bytes": earring_bytes,
                "pose_key": pose_key
            })
            print(f"   [OK] Prepared: {product['name']} (pose: {pose_key})")

        print(f"\n[OK] {len(product_data)} products ready for parallel processing")
        print("="*60 + "\n")

        # ============================================
        # STAGE 1: PARALLEL POSE GENERATION
        # ============================================
        print("="*60)
        print("STAGE 1: GENERATING ALL POSES IN PARALLEL")
        print("="*60)
        stage1_start = time.time()

        async def generate_pose_async(data, sem):
            """Generate a single pose with semaphore control."""
            async with sem:
                product = data["product"]
                print(f"   [START] Generating pose for: {product['name']}")
                try:
                    # Run blocking API call in thread pool
                    pose_bytes = await asyncio.to_thread(
                        vto_generator.generate_styled_pose,
                        face_bytes=front_contents,
                        style_ref_bytes=data["style_bytes"],
                        pose_key=data["pose_key"],
                        session_id=session_id
                    )
                    if pose_bytes:
                        print(f"   [OK] Pose complete: {product['name']}")
                    else:
                        print(f"   [FAIL] Pose failed: {product['name']}")
                    return {"data": data, "pose_bytes": pose_bytes}
                except Exception as e:
                    print(f"   [ERROR] Pose error for {product['name']}: {e}")
                    return {"data": data, "pose_bytes": None, "error": str(e)}

        # Run all pose generations in parallel
        pose_tasks = [generate_pose_async(d, semaphore) for d in product_data]
        pose_results = await asyncio.gather(*pose_tasks)

        stage1_duration = time.time() - stage1_start
        successful_poses = sum(1 for r in pose_results if r.get("pose_bytes"))
        print(f"\n[OK] Stage 1 complete: {successful_poses}/{len(product_data)} poses in {stage1_duration:.2f}s")
        print("="*60 + "\n")

        # ============================================
        # STAGE 2: PARALLEL VTO COMPOSITION
        # ============================================
        print("="*60)
        print("STAGE 2: COMPOSITING ALL EARRINGS IN PARALLEL")
        print("="*60)
        stage2_start = time.time()

        async def generate_vto_async(pose_result, sem):
            """Generate a single VTO with semaphore control."""
            data = pose_result["data"]
            pose_bytes = pose_result.get("pose_bytes")
            product = data["product"]

            if not pose_bytes:
                return {
                    "product_id": product["id"],
                    "product_name": product["name"],
                    "success": False,
                    "error": pose_result.get("error", "Pose generation failed")
                }

            async with sem:
                print(f"   [START] Compositing: {product['name']}")
                try:
                    final_bytes = await asyncio.to_thread(
                        vto_generator.generate_final_vto,
                        pose_bytes=pose_bytes,
                        earring_bytes=data["earring_bytes"],
                        style_bytes=data["style_bytes"],
                        session_id=session_id,
                        product_name=product["name"]
                    )

                    if final_bytes:
                        img_base64 = base64.b64encode(final_bytes).decode('utf-8')
                        print(f"   [OK] VTO complete: {product['name']}")
                        return {
                            "product_id": product["id"],
                            "product_name": product["name"],
                            "success": True,
                            "image": f"data:image/png;base64,{img_base64}"
                        }
                    else:
                        print(f"   [FAIL] VTO failed: {product['name']}")
                        return {
                            "product_id": product["id"],
                            "product_name": product["name"],
                            "success": False,
                            "error": "VTO generation failed - no bytes returned"
                        }
                except Exception as e:
                    print(f"   [ERROR] VTO error for {product['name']}: {e}")
                    return {
                        "product_id": product["id"],
                        "product_name": product["name"],
                        "success": False,
                        "error": str(e)
                    }

        # Run all VTO generations in parallel
        vto_tasks = [generate_vto_async(pr, semaphore) for pr in pose_results]
        results = await asyncio.gather(*vto_tasks)

        stage2_duration = time.time() - stage2_start
        print(f"\n[OK] Stage 2 complete in {stage2_duration:.2f}s")
        print("="*60 + "\n")

        # Summary
        success_count = sum(1 for r in results if r['success'])
        failed_count = len(PRODUCTS) - success_count

        print("\n" + "="*60)
        print("SESSION COMPLETE")
        print("="*60)
        print(f"Session ID: {session_id}")
        print(f"Total Products: {len(PRODUCTS)}")
        print(f"[SUCCESS] Successful: {success_count}")
        print(f"[FAILED] Failed: {failed_count}")
        print(f"[DIR] Output Directory: {session_dir}")

        if success_count > 0:
            print(f"\n[OK] Generated files:")
            for r in results:
                if r['success']:
                    print(f"   • {r['product_name']}")

        if failed_count > 0:
            print(f"\n[X] Failed products:")
            for r in results:
                if not r['success']:
                    print(f"   • {r['product_name']}: {r.get('error', 'Unknown error')}")

        print("="*60 + "\n")

        # Queue background upload to Google Drive (non-blocking)
        if drive_uploader and drive_uploader.enabled and background_tasks:
            background_tasks.add_task(
                upload_session_to_drive,
                uploader=drive_uploader,
                session_id=session_id,
                session_dir=session_dir
            )
            print(f"[Drive] Queued background upload for session: {session_id}")

        # Queue background upload to Cloudinary (non-blocking)
        if cloudinary_uploader and cloudinary_uploader.enabled and background_tasks:
            background_tasks.add_task(
                upload_session_to_cloudinary,
                uploader=cloudinary_uploader,
                session_id=session_id,
                session_dir=session_dir
            )
            print(f"[Cloudinary] Queued background upload for session: {session_id}")

        return {
            "success": True,
            "message": f"Generated {success_count} out of {len(PRODUCTS)} try-on images",
            "session_id": session_id,
            "output_directory": str(session_dir),
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "Pandora API - Jewelry Virtual Try-On with Hybrid Architecture"}
