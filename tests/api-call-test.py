import os
import random
import time
from google import genai
from google.genai import types
from PIL import Image

class MultiPoseJewelryGenerator:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        self.model_id = 'gemini-3-pro-image-preview'
        
        self.intermediate_dir = "intermediate_poses"
        self.final_dir = "final_vto_results"
        for d in [self.intermediate_dir, self.final_dir]:
            if not os.path.exists(d): os.makedirs(d)

        self.target_poses = {
            "profile_left": "a 90-degree profile view looking left",
            "three_quarter_right": "a three-quarter view looking slightly right",
            "slightly_up": "a slight upward gaze, exposing the neck and earlobe clearly"
        }

    def _save_image(self, response, save_path):
        try:
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    with open(save_path, "wb") as f:
                        f.write(part.inline_data.data)
                    return True
            return False
        except Exception as e:
            print(f"Extraction Error: {e}")
            return False

    def generate_styled_pose(self, face_path, style_ref_path, pose_key):
        """Stage 1: Identity-Preserving Pose Variation with Style Reference."""
        pose_desc = self.target_poses[pose_key]
        print(f"-> Stage 1: Generating Pose ({pose_key}) using Style Ref: {os.path.basename(style_ref_path)}")
        
        with open(face_path, "rb") as f: face_data = f.read()
        with open(style_ref_path, "rb") as f: style_data = f.read()

        # We feed both the Identity (face) and the Style (blueprint)
        content_parts = [
            types.Part.from_bytes(data=face_data, mime_type="image/jpeg"),   # Identity Reference
            types.Part.from_bytes(data=style_data, mime_type="image/jpeg"),  # Style Reference
            f"TASK: Identity-Preserving Pose Variation. "
            f"SUBJECT: Maintain 100% facial identity of the first image. "
            f"STYLE: Match the lighting, skin texture, and professional quality of the second image. "
            f"POSE: Re-imagine the person in {pose_desc}. "
            f"CRITICAL: Expose the ear clearly for jewelry placement."
        ]

        save_path = os.path.join(self.intermediate_dir, f"pose_{pose_key}_{int(time.time())}.png")
        
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=content_parts
            )
            if self._save_image(response, save_path):
                return save_path
        except Exception as e:
            print(f"Stage 1 Error: {e}")
        return None

    def generate_final_vto(self, pose_path, earring_path, style_path):
        """Stage 2: Composite the earring onto the styled pose."""
        print(f"-> Stage 2: Final VTO Composition...")
        
        with open(pose_path, "rb") as f: pose_bytes = f.read()
        with open(earring_path, "rb") as f: earring_bytes = f.read()
        with open(style_path, "rb") as f: style_bytes = f.read()

        content_parts = [
            types.Part.from_bytes(data=pose_bytes, mime_type="image/png"),   # Reference 1
            types.Part.from_bytes(data=earring_bytes, mime_type="image/png"),# Reference 2
            types.Part.from_bytes(data=style_bytes, mime_type="image/jpeg"), # Reference 3
            "INSTRUCTION: Composite the EXACT earring from Reference 2 onto the subject in Reference 1. "
            "Maintain the earring identity perfectly. Match lighting to Reference 3."
        ]

        output_path = os.path.join(self.final_dir, f"vto_styled_{int(time.time())}.png")
        
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=content_parts
            )
            if self._save_image(response, output_path):
                print(f"✅ SUCCESS: {output_path}")
                return output_path
        except Exception as e:
            print(f"Stage 2 Error: {e}")
        return None

    def run_workflow(self, face, earring, style_refs):
        """Optimized workflow using a random pose and a random style reference."""
        # 1. Randomly pick ONE pose and ONE style photo from your 3 references
        selected_pose_key = random.choice(list(self.target_poses.keys()))
        selected_style = random.choice(style_refs)
        
        # 2. Generate the styled pose
        pose_img = self.generate_styled_pose(face, selected_style, selected_pose_key)
        
        # 3. Perform final VTO
        if pose_img:
            return self.generate_final_vto(pose_img, earring, selected_style)
        return None

# =========================================
# EXECUTION (Configured for your files)
# =========================================
if __name__ == "__main__":
    # Load API key from environment variable
    import os
    from dotenv import load_dotenv
    load_dotenv()

    API_KEY = os.getenv("GEMINI_API_KEY")
    if not API_KEY:
        print("❌ Error: GEMINI_API_KEY not found in environment variables")
        print("Please set it in your .env file or environment")
        exit(1)
    
    # Primary Inputs
    face_file = "model_test.jpg"
    earring_file = "silver-flower-scaled_earring_output.png"
    
    # ⚠️ List of your 3 Styling Reference photos
    styling_references = [
        "red-heart-2.jpg",        # Reference A
        "model-silver-earing.jpg",  # Reference B (update with your actual filenames)
        "nodel-3-reference.jpg"    # Reference C (update with your actual filenames)
    ]

    # Ensure all files exist before running
    missing = [f for f in [face_file, earring_file] + styling_references if not os.path.exists(f)]
    if missing:
        print(f"❌ Error: Files not found: {missing}")
    else:
        generator = MultiPoseJewelryGenerator(api_key=API_KEY)
        
        # This will run ONCE and pick a random pose and a random style from your list
        generator.run_workflow(face_file, earring_file, styling_references)