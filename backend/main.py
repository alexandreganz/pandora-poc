import os
import sys
import base64
import random
import time
import uuid
import io
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from google import genai
from google.genai import types

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

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
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

        # Target poses for variation
        self.target_poses = {
            "profile_left": "a 90-degree profile view looking left",
            "three_quarter_right": "a three-quarter view looking slightly right",
            "slightly_up": "a slight upward gaze, exposing the neck and earlobe clearly",
            "high_angle_down": "a high-angle view looking down at the subject, emphasizing the top of the head and the bridge of the nose",
            "profile_right_tilt": "a 90-degree profile view looking right with the head tilted slightly toward the shoulder",
            "back_three_quarter_left": "an over-the-shoulder view from behind, looking toward the left to show the jawline and back of the ear"
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
            f"TASK: Identity-Preserving Pose Variation. "
            f"SUBJECT: Maintain 100% facial identity of the first image. "
            f"STYLE: Match the lighting, skin texture, and professional quality of the second image. "
            f"POSE: Re-imagine the person in {pose_desc}. "
            f"CRITICAL: Expose the ear clearly for jewelry placement."
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
            f"TASK: Place the earring from Image 2 onto the person in Image 1. "
            f"EARRING: Use ONLY the exact earring design from Image 2 - this is a {product_name} earring. "
            f"PLACEMENT: Place exactly ONE earring on the visible earlobe. Do NOT add extra earrings. "
            f"IMPORTANT: Do NOT modify, duplicate, or hallucinate different earring designs. "
            f"LIGHTING: Match the lighting style from Image 3. "
            f"OUTPUT: Photorealistic result with the person wearing the single earring naturally."
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
    front_photo: UploadFile = File(...)
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
            print(f"[OK] PPM calculated: {ppm:.4f} pixels/mm")
        else:
            print("[WARNING] PPM calculation failed - using default scaling")
            ppm = None
        print("="*60 + "\n")

        # ============================================
        # GENERATE UNIQUE POSE + COMPOSITION FOR EACH PRODUCT
        # ============================================
        # Shuffle pose keys for variety across products
        pose_keys = list(vto_generator.target_poses.keys())
        random.shuffle(pose_keys)

        print("\n" + "="*60)
        print("GENERATING TRY-ONS WITH UNIQUE POSES")
        print("="*60)
        results = []
        total_products = len(PRODUCTS)

        for idx, product in enumerate(PRODUCTS, 1):
            print(f"\n[{idx}/{total_products}] Processing: {product['name']} (ID: {product['id']})")

            # ============================================
            # Load product's image2 as style reference
            # ============================================
            style_ref_filename = product["image2"].replace("/images/", "")
            style_ref_path = ASSETS_PATH / style_ref_filename

            if style_ref_path.exists():
                with open(style_ref_path, "rb") as f:
                    style_bytes = f.read()
                print(f"   [OK] Style reference loaded: {style_ref_filename}")
            else:
                # Fallback to uploaded photo if image2 not found
                style_bytes = front_contents
                print(f"   [WARNING] Style ref not found, using uploaded photo")

            # ============================================
            # Select pose for this product (cycling through shuffled poses)
            # ============================================
            pose_key = pose_keys[idx % len(pose_keys)]
            print(f"   → Pose: {pose_key}")

            # ============================================
            # STAGE 1: Generate unique pose for this product
            # ============================================
            print(f"   → Generating unique pose variation...")
            stage1_start = time.time()

            pose_bytes = vto_generator.generate_styled_pose(
                face_bytes=front_contents,
                style_ref_bytes=style_bytes,
                pose_key=pose_key,
                session_id=session_id
            )
            stage1_duration = time.time() - stage1_start

            if not pose_bytes:
                print(f"   [ERROR] Pose generation failed for {product['name']}")
                results.append({
                    "product_id": product["id"],
                    "product_name": product["name"],
                    "success": False,
                    "error": "Pose generation failed"
                })
                continue

            print(f"   [OK] Pose generated in {stage1_duration:.2f}s")

            # ============================================
            # Load earring asset
            # ============================================
            print(f"   Loading earring asset: {product['asset']}")
            earring_asset_path = ASSETS_PATH / product["asset"]

            if not earring_asset_path.exists():
                print(f"   [ERROR] Earring asset not found at {earring_asset_path}")
                results.append({
                    "product_id": product["id"],
                    "product_name": product["name"],
                    "success": False,
                    "error": "Earring asset not found"
                })
                continue

            # Read earring asset
            print(f"   [OK] Earring asset loaded: {earring_asset_path}")
            with open(earring_asset_path, "rb") as f:
                earring_bytes = f.read()
            print(f"   [OK] Earring size: {len(earring_bytes)} bytes")

            # Scale earring based on PPM if available
            if ppm and "height_mm" in product and "width_mm" in product:
                print(f"   → Scaling earring to match user's photo...")
                earring_bytes = photo_scaler.scale_earring(
                    earring_bytes=earring_bytes,
                    height_mm=product["height_mm"],
                    width_mm=product["width_mm"],
                    ppm=ppm
                )

            # ============================================
            # STAGE 2: Composite earring onto the unique pose
            # ============================================
            try:
                print(f"   → Sending to Gemini API for composition...")
                stage2_start = time.time()

                final_image_bytes = vto_generator.generate_final_vto(
                    pose_bytes=pose_bytes,
                    earring_bytes=earring_bytes,
                    style_bytes=style_bytes,
                    session_id=session_id,
                    product_name=product["name"]
                )

                stage2_duration = time.time() - stage2_start

                if final_image_bytes:
                    # Convert to base64
                    img_base64 = base64.b64encode(final_image_bytes).decode('utf-8')

                    print(f"   [SUCCESS] IMAGE {idx}/{total_products} GENERATED in {stage2_duration:.2f}s")
                    print(f"   [OK] Final image size: {len(final_image_bytes)} bytes")
                    print(f"   [OK] Base64 encoded: {len(img_base64)} characters")

                    results.append({
                        "product_id": product["id"],
                        "product_name": product["name"],
                        "success": True,
                        "image": f"data:image/png;base64,{img_base64}"
                    })
                else:
                    print(f"   [ERROR] IMAGE {idx}/{total_products} FAILED - No image bytes returned")
                    results.append({
                        "product_id": product["id"],
                        "product_name": product["name"],
                        "success": False,
                        "error": "Image generation failed - no bytes returned"
                    })

            except Exception as e:
                print(f"   [ERROR] IMAGE {idx}/{total_products} ERROR: {type(e).__name__}: {str(e)}")
                import traceback
                print(f"   Traceback: {traceback.format_exc()}")
                results.append({
                    "product_id": product["id"],
                    "product_name": product["name"],
                    "success": False,
                    "error": str(e)
                })

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
