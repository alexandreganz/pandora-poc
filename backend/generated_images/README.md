# Generated Images Directory

This directory contains all AI-generated virtual try-on images organized by session.

## Directory Structure

```
generated_images/
├── poses/                    # Deprecated: Legacy pose storage
├── try_ons/                  # Deprecated: Legacy try-on storage
├── <session_id>/             # Session-specific directory (timestamp_uuid)
│   ├── original_photo.jpg    # User's uploaded front-facing photo
│   ├── <session_id>_pose_<pose_type>_<timestamp>.png    # Generated pose variation
│   ├── <session_id>_classic_circle_<timestamp>.png      # Try-on: Classic Circle
│   ├── <session_id>_silver_flower_<timestamp>.png       # Try-on: Silver Flower
│   ├── <session_id>_red_heart_<timestamp>.png           # Try-on: Red Heart
│   ├── <session_id>_gold_heart_<timestamp>.png          # Try-on: Gold Heart
│   └── <session_id>_silver_heart_<timestamp>.png        # Try-on: Silver Heart
└── README.md                 # This file

```

## Session Directory Format

Each try-on session creates a unique directory with format:
- `YYYYMMDD_HHMMSS_<8-char-uuid>`
- Example: `20260113_143022_a7b3c8d9`

## Files in Each Session

### 1. Original Photo
- **Filename**: `original_photo.jpg`
- **Description**: The user's uploaded front-facing photo

### 2. Pose Variation
- **Filename**: `<session_id>_pose_<pose_type>_<timestamp>.png`
- **Description**: AI-generated pose variation (Stage 1)
- **Pose Types**:
  - `profile_left`: 90-degree profile view looking left
  - `three_quarter_right`: Three-quarter view looking slightly right
  - `slightly_up`: Slight upward gaze

### 3. Try-On Results (5 Files)
- **Filename**: `<session_id>_<product_name>_<timestamp>.png`
- **Description**: Final virtual try-on images (Stage 2)
- **Products**:
  1. Classic Circle
  2. Silver Flower
  3. Red Heart
  4. Gold Heart
  5. Silver Heart

## Workflow

1. **Upload**: User uploads front-facing photo
2. **Stage 1**: Generate single pose variation
3. **Stage 2**: Composite each of 5 earrings onto pose
4. **Save**: All images saved in session directory
5. **Return**: Base64-encoded images returned to frontend

## Storage Notes

- Images are saved in PNG format (except original which is JPEG)
- All session directories persist until manually deleted
- To clean up old sessions, delete session directories
- Keep this README file for documentation

## API Response

The `/api/try-on-all` endpoint returns:
```json
{
  "success": true,
  "session_id": "20260113_143022_a7b3c8d9",
  "output_directory": "/path/to/generated_images/20260113_143022_a7b3c8d9",
  "results": [...]
}
```

## Maintenance

### Cleanup Old Sessions
```bash
# Delete sessions older than 30 days
find generated_images/ -maxdepth 1 -type d -mtime +30 -exec rm -rf {} \;
```

### Monitor Disk Usage
```bash
# Check total size
du -sh generated_images/

# List sessions by size
du -sh generated_images/*/ | sort -h
```
