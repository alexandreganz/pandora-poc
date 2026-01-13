import cv2
import mediapipe as mp
from PIL import Image
import os

def run_scaling_pipeline(model_photo_path, product_png_path, tech_h_mm, tech_w_mm, output_path="scaled_earring_output.png"):
    # --- STEP 1: IRIS DETECTION ---
    mp_face_mesh = mp.solutions.face_mesh
    HVID_MM = 11.7  # Standard iris diameter constant

    image = cv2.imread(model_photo_path)
    if image is None:
        print("Error: Could not find your model photo. Check the file name.")
        return
    
    h, w, _ = image.shape
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:

        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            print("Error: Face detection failed. Ensure your eyes are visible in the photo.")
            return

        # Indices for Right Iris
        mesh_points = results.multi_face_landmarks[0].landmark
        p1 = mesh_points[474] # Left side of right iris
        p2 = mesh_points[476] # Right side of right iris
        
        # Calculate horizontal pixel distance
        dist_px = (( (p2.x - p1.x) * w)**2 + ( (p2.y - p1.y) * h)**2)**0.5
        ppm = dist_px / HVID_MM
        
        print(f"--- STEP 1: ANALYZING YOUR PHOTO ---")
        print(f"Iris width in pixels: {dist_px:.2f}")
        print(f"Scale Factor: {ppm:.4f} pixels per mm")

    # --- STEP 2: PRECISION PRODUCT SCALING ---
    if not os.path.exists(product_png_path):
        print(f"Error: Product file '{product_png_path}' not found.")
        return

    with Image.open(product_png_path) as img:
        # Calculate target dimensions in pixels
        target_h_px = int(tech_h_mm * ppm)
        target_w_px = int(tech_w_mm * ppm)
        
        # Resize using Lanczos (best for maintaining detail in small items)
        resized_product = img.resize((target_w_px, target_h_px), Image.Resampling.LANCZOS)
        resized_product.save(output_path)
        
        print(f"\n--- STEP 2: SCALING PRODUCT ---")
        print(f"Targeting Tech Specs: {tech_w_mm}mm x {tech_h_mm}mm")
        print(f"Resulting Pixel Size: {target_w_px}px x {target_h_px}px")
        print(f"SUCCESS: Scaled PNG saved as '{output_path}'")

# --- USER INPUT SECTION ---
# Update these with your specific filenames
MY_PHOTO = "model-test-2.jpg" 
EARRING_FILE = "silver-flower-transparent.png"

# Your specific dimensions provided:
E_HEIGHT = 6.7 
E_WIDTH = 6.8

run_scaling_pipeline(MY_PHOTO, EARRING_FILE, E_HEIGHT, E_WIDTH)