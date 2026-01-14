# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pandora is a luxury jewelry online store with AI-powered virtual try-on capabilities. Features a high-end aesthetic with elegant typography and warm gold accents.

### Features
- 6 purchasable jewelry items (earrings with known dimensions in mm)
- **Luxury UI**: Elegant design with Cormorant Garamond & Montserrat fonts, 4:5 aspect ratio cards, 3-column grid
- **Image Carousel**: Interactive carousel on each product card
  - Shows 2 product images by default (different angles)
  - Navigation via arrow buttons (prev/next)
  - Dot indicators for current image
  - Automatically adds a 3rd image after user completes virtual try-on
- **Virtual Try-On**: Two-stage AI workflow for realistic earring preview
- Try-on results persist in product carousels for easy comparison

### Products
1. Classic Circle ($89) - 25mm x 25mm
2. Silver Flower ($95) - 6.7mm x 6.8mm
3. Red Heart ($79) - 15mm x 14mm
4. Gold Heart ($125) - 18mm x 16mm
5. Silver Heart ($85) - 15mm x 14mm
6. Blue Butterfly ($110) - 20mm x 22mm

## Virtual Try-On Feature

### Two-Stage AI Workflow

**Stage 1: Identity-Preserving Pose Generation**
- Takes user's front-facing photo + style reference (product image2)
- Generates unique pose variation (6 pose types available)
- CRITICAL: Preserves 100% of user's physical characteristics
- Style reference used ONLY for lighting and positioning

**Stage 2: Earring Composition**
- Takes posed image + earring asset + style reference
- Composites exact earring onto the pose
- Maintains user identity and earring design integrity
- Uses style reference ONLY for lighting/shadows

### Pose Variations (6 types)
- `profile_left`: 90-degree profile view looking left
- `three_quarter_right`: Three-quarter view looking slightly right
- `slightly_up`: Slight upward gaze, exposing neck and earlobe
- `high_angle_down`: High-angle view looking down
- `profile_right_tilt`: 90-degree profile right with head tilt
- `back_three_quarter_left`: Over-the-shoulder view from behind

### Identity Preservation (Prompt Engineering)
The prompts explicitly preserve:
- Face shape, eyes, nose, mouth, eyebrows
- Skin tone, texture, complexion
- Hair color, style, length, texture
- Age and ethnic features
- States: "DO NOT borrow ANY physical features from reference images"

## Tech Stack

- **Frontend**: React 19 + Vite 7
- **Backend**: Python 3.12 + FastAPI
- **Computer Vision**: MediaPipe (face mesh, iris detection)
- **AI Image Generation**: Google Gemini API (gemini-3-pro-image-preview)
- **Styling**: CSS with Google Fonts (Cormorant Garamond, Montserrat)

## Build Commands

### Frontend
```bash
cd frontend
npm install        # Install dependencies
npm run dev        # Start dev server (http://localhost:5173 or 5174)
npm run build      # Build for production
npm run lint       # Run ESLint
```

### Backend
```bash
cd backend
pip install -r requirements.txt   # Install dependencies
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Docker (Local Development)
```bash
docker-compose up       # Start both frontend and backend
docker-compose down     # Stop services
```

**Key Backend Features:**
- Iris-based scale calibration (11.7mm HVID biological constant)
- MediaPipe face mesh for landmark detection
- PIL with LANCZOS resampling for high-quality earring scaling
- Session-based image storage (organized by timestamp)
- Environment-based CORS configuration

## Architecture

```
pandora/
├── frontend/                    # React + Vite app
│   ├── src/
│   │   ├── App.jsx             # Main app component
│   │   ├── App.css             # Component styles
│   │   └── index.css           # Global styles + CSS variables
│   ├── public/images/          # Product images
│   ├── Dockerfile              # Production container
│   ├── nginx.conf              # Production server config
│   └── vercel.json             # Vercel deployment config
├── backend/
│   ├── main.py                 # FastAPI server + AI logic
│   ├── requirements.txt        # Python dependencies
│   ├── generated_images/       # Session output folders
│   │   └── {session_id}/
│   │       ├── original_photo.jpg
│   │       ├── poses/          # Stage 1 outputs
│   │       └── try_ons/        # Stage 2 outputs
│   ├── Dockerfile              # Production container
│   ├── railway.json            # Railway deployment config
│   └── .env.example            # Environment template
└── docker-compose.yml          # Local containerized testing
```

### API Endpoints
- `GET /api/products` - Returns list of 6 jewelry items with name, price, dimensions, images
- `POST /api/try-on-all` - Accepts front photo; returns 6 generated try-on images (one per product)

## Key Code Locations

### Prompt Engineering (backend/main.py)
| Stage | Lines | Purpose |
|-------|-------|---------|
| Stage 1 | ~320-340 | Pose generation with identity preservation |
| Stage 2 | ~396-420 | Earring composition |

### Other Key Classes (backend/main.py)
| Class | Lines | Purpose |
|-------|-------|---------|
| PhotoScaler | ~131-237 | Iris detection and PPM calculation for earring scaling |
| VirtualTryOnGenerator | ~240-479 | Two-stage AI workflow orchestration |

### Product Data (backend/main.py)
- Lines ~64-128: PRODUCTS array with id, name, price, images, dimensions

### API URL Configuration
- `frontend/src/App.jsx` line ~9: `API_URL` constant (update for deployment)

### Frontend Styling
- `frontend/src/index.css`: CSS variables, fonts, global styles
- `frontend/src/App.css`: Component styles, grid layout, animations

### Test Files
- `tests/api-call-test.py`: API testing for virtual try-on workflow
- `tests/photo_scaling.py`: Standalone iris detection and scaling tests

## Development Workflow

### Auto-Reload Behavior
- **Backend**: Running with `--reload` flag auto-restarts on file changes
- **Frontend**: Vite HMR (Hot Module Replacement) for instant CSS/JS updates

### Testing Changes
1. Make changes to code
2. Save file
3. Wait 2-3 seconds for reload
4. Refresh browser / test feature
5. Check backend logs: `tail -50 {output_file}`

### Session Image Storage
Generated images are organized by session:
```
generated_images/
└── 20260114_123456_abc12345/   # Session folder (timestamp_uuid)
    ├── original_photo.jpg       # User's uploaded photo
    ├── poses/                   # Stage 1 outputs
    │   ├── pose_profile_left_xxx.png
    │   └── pose_three_quarter_right_xxx.png
    └── try_ons/                 # Stage 2 outputs
        ├── classic_circle_xxx.png
        └── blue_butterfly_xxx.png
```

## Deployment

### Environment Variables
```env
GEMINI_API_KEY=your_gemini_api_key_here
CORS_ORIGINS=https://your-frontend.vercel.app
```

### Recommended Stack
- **Frontend**: Vercel (free tier, auto-deploy from GitHub)
- **Backend**: Railway (~$5/mo, Docker support)

### Deployment Files
- `frontend/vercel.json` - Vercel configuration
- `backend/railway.json` - Railway configuration
- `backend/Dockerfile` - Python 3.12 + OpenCV/MediaPipe
- `frontend/Dockerfile` - Multi-stage Node → Nginx build
- `docker-compose.yml` - Local containerized testing

### Deploy Steps
1. **Frontend → Vercel**: Import GitHub repo, set root to `frontend`
2. **Backend → Railway**: Import repo, set root to `backend`, add env vars
3. Update `API_URL` in `frontend/src/App.jsx` to Railway URL
4. Update `CORS_ORIGINS` in Railway to Vercel domain

## Processing Flow

```
Frontend                           Backend
   │                                  │
   ├── Upload front photo ───────────►│
   │                                  │
   │                                  ├── Save original photo
   │                                  ├── Calculate PPM (iris detection)
   │                                  │
   │                                  │ FOR EACH PRODUCT (6x):
   │                                  │   │
   │                                  │   ├── Load product's image2 as style ref
   │                                  │   ├── Select random pose type
   │                                  │   │
   │                                  │   ├── STAGE 1: Generate pose
   │                                  │   │   └── Preserve user identity 100%
   │                                  │   │   └── Use style ref for lighting only
   │                                  │   │   └── Save to poses/ folder
   │                                  │   │
   │                                  │   ├── Scale earring asset (PPM × mm)
   │                                  │   │
   │                                  │   ├── STAGE 2: Composite earring
   │                                  │   │   └── Keep user identity from Stage 1
   │                                  │   │   └── Use exact earring design
   │                                  │   │   └── Save to try_ons/ folder
   │                                  │   │
   │◄─────────────────────────────────┤   └── Return base64 image
   │                                  │
   ├── Store in tryOnResults state    │
   ├── Add to product carousel        │
   └── User browses results           │
```
