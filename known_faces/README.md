# Known Faces Database - Reference Photo Guide
# =============================================

## How This Improves Recognition

The known faces database provides **reference photos** that the system uses as 
ground truth for face recognition. This is how professional face recognition 
systems (like the Chinese systems you mentioned) work:

1. **Enrollment**: High-quality photos taken in controlled conditions
2. **Reference Matching**: Live detections are compared against these references FIRST
3. **Consistency**: Known people always get the same ID, no matter which camera

## Directory Structure

```
known_faces/
├── person_001/
│   ├── photo_01.jpg      # Front-facing photo
│   ├── photo_02.jpg      # Slight left angle
│   └── photo_03.jpg      # Slight right angle
├── person_002/
│   ├── photo_01.jpg
│   └── photo_02.jpg
└── ...

known_faces.json          # Metadata (names, notes, etc.)
```

## How to Add Reference Photos

### Method 1: Via GUI (Recommended)
1. Wait for the person to appear in the camera
2. Click on their card in the sidebar
3. Enter their first and last name
4. Click "Dodaj Klienta" 
5. Their face will be saved to the database automatically

### Method 2: Upload Photo via GUI
1. Click the 📷 button in the sidebar
2. Select a high-quality photo of the person
3. If recognized, they'll be matched; if new, assign a name

### Method 3: Manual File Addition
1. Create a folder: `known_faces/person_XXX/`
2. Add high-quality face photos (JPG/PNG)
3. Restart the application - it will auto-discover new folders
4. (Optional) Edit `known_faces.json` to add names

## Best Practices for Reference Photos

### Photo Quality
- ✅ High resolution (at least 200x200 face area)
- ✅ Well-lit, even lighting
- ✅ Clear, in-focus
- ✅ Face occupies most of the frame
- ❌ No sunglasses or heavy shadows
- ❌ No extreme angles (profile views)
- ❌ No motion blur

### Multiple Photos Per Person
For best recognition, add 3-5 photos per person:
- Front-facing (main reference)
- Slight left turn (~15-30°)
- Slight right turn (~15-30°)
- Different lighting conditions if possible
- Different days/times (hair, accessories may vary)

### Photo Naming Convention
- `photo_01.jpg`, `photo_02.jpg`, etc. (auto-added via GUI)
- `capture_01.jpg`, `capture_02.jpg`, etc. (captured from live feed)
- Any `.jpg`, `.jpeg`, `.png`, `.bmp` will be loaded

## Example known_faces.json

```json
{
  "version": "1.0",
  "updated_at": 1704672000.0,
  "persons": [
    {
      "person_id": "person_001",
      "first_name": "Jan",
      "last_name": "Kowalski",
      "notes": "VIP customer",
      "photo_paths": [
        "known_faces/person_001/photo_01.jpg",
        "known_faces/person_001/photo_02.jpg"
      ],
      "created_at": 1704672000.0,
      "updated_at": 1704672000.0
    },
    {
      "person_id": "person_002", 
      "first_name": "Anna",
      "last_name": "Nowak",
      "notes": "Staff member",
      "photo_paths": [
        "known_faces/person_002/photo_01.jpg"
      ],
      "created_at": 1704672000.0,
      "updated_at": 1704672000.0
    }
  ]
}
```

## Matching Thresholds

The known faces matcher uses stricter thresholds than dynamic matching:
- **KNOWN_MATCH_THRESHOLD**: 0.65 (minimum similarity to match)
- **KNOWN_STRONG_MATCH**: 0.75 (high confidence match)

You can adjust these in `known_faces_manager.py` if needed.

## Troubleshooting

### Person Not Recognized
- Add more reference photos from different angles
- Ensure reference photos are well-lit
- Check if face is too small in camera view

### Wrong Person Matched
- Add more diverse reference photos for both people
- Consider increasing KNOWN_MATCH_THRESHOLD
- Ensure photos don't include other people's faces

### No Faces Detected in Uploaded Photo
- Ensure face is visible and not too small
- Check image is not corrupted
- Try a clearer, front-facing photo
