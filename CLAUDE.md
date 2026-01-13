# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pandora is a minimalist, modern jewelry online store with synchronized backend and frontend.

### Features
- 5 purchasable jewelry items (earrings with known dimensions)
- **Image Carousel**: Interactive carousel on each product card
  - Shows 2 product images by default (different angles)
  - Navigation via arrow buttons (prev/next)
  - Dot indicators for current image
  - Automatically adds a 3rd image after user completes virtual try-on
- Each product card shows item name and price below the image
- **Virtual Try-On**: AI-powered earring preview on user photos
- Try-on results persist in product carousels for easy comparison

### Virtual Try-On Feature
Users upload a single front-facing photo to virtually try on earrings:

1. **Front photo** (facing camera directly with both ears visible)
   - MediaPipe extracts face mesh landmarks
   - Calculate iris diameter to determine scale (pixels per mm)
   - Detect both earlobe positions for earring placement

2. **AI Generation** (Google Gemini Imagen)
   - Combine user face measurements + earring dimensions
   - Generate realistic try-on image with correctly sized earrings on both ears

3. **Carousel Integration**
   - Try-on result is automatically added as the 3rd image in the product's carousel
   - Users can return to the main page and browse their try-on results
   - Each product remembers its try-on image for the session

## User Journey

1. **Browse Products**: User sees 5 earring products, each with a 2-image carousel
2. **Virtual Try-On**: User clicks "Virtual Try-On" button in header
3. **Upload Photo**: User uploads a single front-facing photo
4. **Select Earrings**: User clicks on earrings to try them on
5. **View Results**: AI-generated try-on appears in modal
6. **Return to Shopping**: User closes modal and returns to product grid
7. **Browse Try-Ons**: Each tried-on product now has 3 images in its carousel
8. **Compare Options**: User can navigate carousels to compare different earrings on their photo
9. **Add to Cart**: User adds desired items to cart and proceeds to checkout

## Tech Stack

- **Frontend**: React + Vite
- **Backend**: Python + FastAPI
- **Computer Vision**: MediaPipe (face mesh, ear detection)
- **AI Image Generation**: Google Gemini Imagen API
- **Payments**: Mock checkout (simulated)

## Build Commands

### Frontend
```bash
cd frontend
npm install        # Install dependencies
npm run dev        # Start dev server (http://localhost:5173)
npm run build      # Build for production
```

### Backend
```bash
cd backend
pip install -r requirements.txt   # Install dependencies
python -m uvicorn main:app --reload   # Start dev server (http://localhost:8000)
```

**Key Backend Features:**
- Iris-based scale calibration (11.7mm biological constant)
- MediaPipe face mesh for landmark detection
- PIL with LANCZOS resampling for high-quality earring scaling
- Dual earlobe detection (both ears from single photo)
- Gemini 2.0 Flash for photorealistic AI blending

## Architecture

```
pandora/
├── frontend/          # React + Vite app
│   ├── src/
│   │   ├── components/   # React components
│   │   ├── assets/       # Images, fonts
│   │   └── App.jsx       # Main app component
│   └── package.json
└── backend/           # FastAPI server
    ├── main.py           # API routes
    └── requirements.txt
```

### API Endpoints
- `GET /api/products` - Returns list of jewelry items with name, price, dimensions, images
- `POST /api/try-on` - Accepts front photo + selected earring ID; returns generated try-on image with earrings on both ears

### Product Carousel Implementation

Each product card includes an interactive image carousel:

**Default State (No Try-On)**
- Image 1: Primary product photo
- Image 2: Secondary product photo (different angle)

**After Try-On**
- Image 1: Primary product photo
- Image 2: Secondary product photo
- Image 3: User's try-on result (AI-generated)

**Navigation**
- Arrow buttons (prev/next) - appear on hover
- Dot indicators at bottom - click to jump to specific image
- Smooth transitions between images

**State Management**
- Try-on results stored in App component (`tryOnResults` state)
- Passed down to individual ProductCard components
- Persists during session (resets on page refresh)

### Try-On Processing Flow
```
Frontend                    Backend
   │                           │
   ├── Upload front photo ────►│
   │   + earring ID            │
   │                           ├── MediaPipe: extract face landmarks
   │                           ├── Calculate iris diameter → pixels per mm (PPM)
   │                           ├── Detect both earlobe positions
   │                           ├── Get earring dimensions from DB
   │                           ├── Scale earring to correct size (dimensions × PPM)
   │                           ├── Place earrings on both ears
   │                           ├── Call Gemini Imagen API for realistic blending
   │                           │
   │◄── Generated image ───────┤
   │
   ├── Store try-on image
   │   in App state
   │
   ├── Add image to product
   │   carousel (3rd image)
   │
   └── User can navigate
       carousel to view
       try-on result
```
