import os
import base64
import random
import time
import uuid
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

app = FastAPI(title="Pandora API", version="1.0.0")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini with new SDK
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_ID = 'gemini-2.0-flash-exp'

# Path to earring assets (transparent PNGs)
ASSETS_PATH = Path(__file__).parent.parent / "frontend" / "public" / "images"

# Output directories for generated images
OUTPUT_BASE_PATH = Path(__file__).parent / "generated_images"
POSES_OUTPUT_PATH = OUTPUT_BASE_PATH / "poses"
TRYON_OUTPUT_PATH = OUTPUT_BASE_PATH / "try_ons"

# Create output directories if they don't exist
for path in [OUTPUT_BASE_PATH, POSES_OUTPUT_PATH, TRYON_OUTPUT_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# Product data
PRODUCTS = [
    {
        "id": 1,
        "name": "Classic Circle",
        "price": 89.00,
        "image1": "/images/classic-circle-1.jpg",
        "image2": "/images/classic-circle-2.jpg",
        "asset": "classic-circle-1.jpg"
    },
    {
        "id": 2,
        "name": "Silver Flower",
        "price": 95.00,
        "image1": "/images/silver-flower-1.jpg",
        "image2": "/images/silver-flower-2.jpg",
        "asset": "silver-flower-1.jpg"
    },
    {
        "id": 3,
        "name": "Red Heart",
        "price": 79.00,
        "image1": "/images/red-heart-1.jpg",
        "image2": "/images/red-heart-2.jpg",
        "asset": "red-heart-1.jpg"
    },
    {
        "id": 4,
        "name": "Gold Heart",
        "price": 125.00,
        "image1": "/images/gold-heart-1.jpg",
        "image2": "/images/gold-heart-2.jpg",
        "asset": "gold-heart-1.jpg"
    },
    {
        "id": 5,
        "name": "Silver Heart",
        "price": 85.00,
        "image1": "/images/silver-heart-1.jpg",
        "image2": "/images/silver-heart-2.jpg",
        "asset": "silver-heart-1.jpg"
    }
]


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
            "slightly_up": "a slight upward gaze, exposing the neck and earlobe clearly"
        }

    def _save_image_from_response(self, response) -> bytes:
        """Extract image bytes from Gemini response."""
        try:
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    return part.inline_data.data
            return None
        except Exception as e:
            print(f"Image extraction error: {e}")
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
                contents=content_parts
            )
            print(f"   ✓ Gemini API responded")

            pose_bytes = self._save_image_from_response(response)

            if not pose_bytes:
                print(f"   ❌ ERROR: Failed to extract image from Gemini response")
                return None

            print(f"   ✓ Extracted image: {len(pose_bytes)} bytes")

            # Save pose image to disk
            if pose_bytes and session_id:
                filename = f"{session_id}_pose_{pose_key}_{int(time.time())}.png"
                filepath = self.poses_dir / filename
                with open(filepath, "wb") as f:
                    f.write(pose_bytes)
                print(f"   ✓ Saved pose to: {filepath}")

            return pose_bytes
        except Exception as e:
            print(f"   ❌ STAGE 1 EXCEPTION: {type(e).__name__}: {str(e)}")
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
            types.Part.from_bytes(data=pose_bytes, mime_type="image/png"),   # Reference 1
            types.Part.from_bytes(data=earring_bytes, mime_type="image/png"),# Reference 2
            types.Part.from_bytes(data=style_bytes, mime_type="image/jpeg"), # Reference 3
            "INSTRUCTION: Composite the EXACT earring from Reference 2 onto the subject in Reference 1. "
            "Maintain the earring identity perfectly. Match lighting to Reference 3."
        ]

        try:
            print(f"      → Calling Gemini API ({self.model_id})...")
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=content_parts
            )
            print(f"      ✓ Gemini API responded")

            final_bytes = self._save_image_from_response(response)

            if not final_bytes:
                print(f"      ❌ ERROR: Failed to extract image from Gemini response")
                return None

            print(f"      ✓ Extracted final image: {len(final_bytes)} bytes")

            # Save final try-on image to disk
            if final_bytes and session_id:
                safe_product_name = product_name.replace(" ", "_").lower() if product_name else "unknown"
                filename = f"{session_id}_{safe_product_name}_{int(time.time())}.png"
                filepath = self.tryon_dir / filename
                with open(filepath, "wb") as f:
                    f.write(final_bytes)
                print(f"      ✓ Saved to disk: {filepath}")

            return final_bytes
        except Exception as e:
            print(f"      ❌ STAGE 2 EXCEPTION: {type(e).__name__}: {str(e)}")
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
        print(f"✓ Saved original photo: {original_photo_path}")

        # Load style references from assets
        # These should be professionally-lit jewelry model photos
        style_refs = []
        style_ref_filenames = [
            "red-heart-2.jpg",
            "silver-heart-1.jpg",
            "gold-heart-1.jpg"
        ]

        for filename in style_ref_filenames:
            style_path = ASSETS_PATH / filename
            if style_path.exists():
                with open(style_path, "rb") as f:
                    style_refs.append(f.read())

        # If no style references found, use the uploaded photo as style reference
        if not style_refs:
            style_refs = [front_contents]

        # Select a random style reference for consistency across all earrings
        selected_style = random.choice(style_refs)

        # ============================================
        # STAGE 1: Generate ONE pose variation (used for all earrings)
        # ============================================
        print("\n" + "="*60)
        print("STAGE 1: POSE GENERATION")
        print("="*60)
        print(f"Generating single pose variation for all earrings...")
        print(f"Using style reference, sending to Gemini API...")

        stage1_start = time.time()
        pose_bytes = vto_generator.generate_styled_pose(
            face_bytes=front_contents,
            style_ref_bytes=selected_style,
            session_id=session_id
        )
        stage1_duration = time.time() - stage1_start

        if not pose_bytes:
            print(f"❌ STAGE 1 FAILED - No pose bytes returned")
            raise HTTPException(
                status_code=500,
                detail="Failed to generate pose variation"
            )

        print(f"✅ STAGE 1 COMPLETE - Pose generated in {stage1_duration:.2f} seconds")
        print(f"   Pose image size: {len(pose_bytes)} bytes")
        print("="*60 + "\n")

        # ============================================
        # STAGE 2: Composite each earring onto the SAME pose
        # ============================================
        print("\n" + "="*60)
        print("STAGE 2: EARRING COMPOSITION")
        print("="*60)
        results = []
        total_products = len(PRODUCTS)

        for idx, product in enumerate(PRODUCTS, 1):
            print(f"\n[{idx}/{total_products}] Processing: {product['name']} (ID: {product['id']})")
            print(f"   Loading earring asset: {product['asset']}")

            # Load earring asset
            earring_asset_path = ASSETS_PATH / product["asset"]

            if not earring_asset_path.exists():
                print(f"   ❌ ERROR: Earring asset not found at {earring_asset_path}")
                results.append({
                    "product_id": product["id"],
                    "product_name": product["name"],
                    "success": False,
                    "error": "Earring asset not found"
                })
                continue

            # Read earring asset
            print(f"   ✓ Earring asset loaded: {earring_asset_path}")
            with open(earring_asset_path, "rb") as f:
                earring_bytes = f.read()
            print(f"   ✓ Earring size: {len(earring_bytes)} bytes")

            # Run Stage 2 only (pose already generated)
            try:
                print(f"   → Sending to Gemini API for composition...")
                stage2_start = time.time()

                final_image_bytes = vto_generator.generate_final_vto(
                    pose_bytes=pose_bytes,
                    earring_bytes=earring_bytes,
                    style_bytes=selected_style,
                    session_id=session_id,
                    product_name=product["name"]
                )

                stage2_duration = time.time() - stage2_start

                if final_image_bytes:
                    # Convert to base64
                    img_base64 = base64.b64encode(final_image_bytes).decode('utf-8')

                    print(f"   ✅ IMAGE {idx}/{total_products} GENERATED in {stage2_duration:.2f}s")
                    print(f"   ✓ Final image size: {len(final_image_bytes)} bytes")
                    print(f"   ✓ Base64 encoded: {len(img_base64)} characters")

                    results.append({
                        "product_id": product["id"],
                        "product_name": product["name"],
                        "success": True,
                        "image": f"data:image/png;base64,{img_base64}"
                    })
                else:
                    print(f"   ❌ IMAGE {idx}/{total_products} FAILED - No image bytes returned")
                    results.append({
                        "product_id": product["id"],
                        "product_name": product["name"],
                        "success": False,
                        "error": "Image generation failed - no bytes returned"
                    })

            except Exception as e:
                print(f"   ❌ IMAGE {idx}/{total_products} ERROR: {type(e).__name__}: {str(e)}")
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
        print(f"✅ Successful: {success_count}")
        print(f"❌ Failed: {failed_count}")
        print(f"📁 Output Directory: {session_dir}")

        if success_count > 0:
            print(f"\n✓ Generated files:")
            for r in results:
                if r['success']:
                    print(f"   • {r['product_name']}")

        if failed_count > 0:
            print(f"\n✗ Failed products:")
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
