# AI Jewelry Virtual Try-On: Hybrid Technical Architecture
This repository contains a Proof of Concept (POC) workflow designed to solve the scaling and realism problem in digital jewelry try-ons. By combining Python-based anatomical measurement with Nano Banana Pro (Gemini 3), we ensure that jewelry is rendered at its actual physical size while maintaining photorealistic lighting.

## Project Overview
The workflow operates in two distinct phases to eliminate "AI Hallucination" regarding product dimensions.

## Phase 1: Mathematical Pre-Processing (Python)
Goal: To establish a "ground truth" for scale and position.
### 1.1 Iris-Based Calibration
Instead of requiring a physical reference object (like a credit card), we use the human iris as a biological ruler.
- Metric: The average adult human iris is consistently 11.7mm in diameter.
- Logic: 
    1. Detect iris landmarks using MediaPipe Iris.
    2. Calculate the pixel width of the iris in a front-facing photo.
    3. Determine the Pixels-Per-Millimeter ($PPM$) ratio.

### 1.2 Ear Landmark Detection
- Using the Diagonal Photo, the script identifies the specific coordinates for the earlobe (Landmarks #177 and #132).
- This provides the X and Y "anchor points" for placement.
### 1.3 Geometric Asset Scaling
- The script takes the 2D earring asset and its real-world height (e.g., 6.3mm).
- It calculates the target pixel height: $Target_{px} = 6.3 \times PPM$.
- Output: A guide_image.png with the earring asset superimposed at the mathematically correct size.

## Phase 2: Neural Rendering (AI Integration)
Goal: To blend the "pasted" asset into the scene using Nano Banana Pro.

### 2.1 The AI Model
- Model: Gemini 3 (Nano Banana Pro) via Google AI Studio API.
- Mode: Image-to-Image (Img2Img).

### 2.2 Integration Prompting
The prompt is engineered to focus the AI on integration rather than generation.
- Prompt: > "High-end jewelry photography. Blend the provided earring asset onto the earlobe. Render realistic soft shadows on the skin and subsurface scattering on the earlobe. Maintain the exact scale and shape of the earring. Cinematic lighting."

### 2.3 Denoising Constraints
To prevent the AI from resizing the earring (a common failure point), we use a specific Denoising Strength:
- Value: 0.35
- Result: The AI preserves the geometric "Guide" from Phase 1 but re-renders the pixels to match the skin texture and lighting of the client.

