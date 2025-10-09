# Test Plate Images

## Overview

This directory contains test plate images for OCR and color/layer classification testing.

## Required Test Images

### Single-Layer Plates (10 images minimum)
- `single_layer_sample.jpg` - Default single-layer test image
- `single_layer_001.jpg` to `single_layer_010.jpg` - Various single-layer plates

### Double-Layer Plates (10 images minimum)
- `double_layer_sample.jpg` - Default double-layer test image
- `double_layer_001.jpg` to `double_layer_010.jpg` - Various double-layer plates

### Color-Specific Plates (5 images per color)
- Blue plates: `blue_plate.jpg`, `blue_001.jpg` to `blue_005.jpg`
- Yellow plates: `yellow_plate.jpg`, `yellow_001.jpg` to `yellow_005.jpg`
- White plates: `white_plate.jpg`, `white_001.jpg` to `white_005.jpg`
- Black plates: `black_plate.jpg`, `black_001.jpg` to `black_005.jpg`
- Green plates: `green_plate.jpg`, `green_001.jpg` to `green_005.jpg`

## Image Requirements

- **Format**: JPEG or PNG
- **Size**: Approximately 140x440 pixels (typical plate dimensions)
- **Color Space**: BGR (OpenCV default)
- **Quality**: Clear, well-lit images with minimal occlusion

## How to Add Test Images

1. **Copy images to this directory**:
   ```bash
   cp /path/to/your/plates/*.jpg tests/fixtures/plates/
   ```

2. **Rename following the naming convention**:
   ```bash
   mv plate1.jpg single_layer_001.jpg
   mv plate2.jpg double_layer_001.jpg
   ```

3. **Verify image quality**:
   ```python
   import cv2
   img = cv2.imread('tests/fixtures/plates/single_layer_001.jpg')
   print(f"Image shape: {img.shape}")  # Should be (H, W, 3)
   cv2.imshow('Test', img)
   cv2.waitKey(0)
   ```

## Synthetic Images

If real plate images are unavailable, the test framework will automatically generate synthetic images using `conftest._generate_synthetic_plate()`. However, real images are **strongly recommended** for accurate testing.

## Privacy Notice

⚠️ **Do NOT commit real vehicle plate images to version control** ⚠️

- Add real images to `.gitignore`
- Use anonymized or synthetic data for public repositories
- Store real test data in a secure, private location

## Current Status

- [ ] Single-layer plates collected (0/10)
- [ ] Double-layer plates collected (0/10)
- [ ] Blue plates collected (0/5)
- [ ] Yellow plates collected (0/5)
- [ ] White plates collected (0/5)
- [ ] Black plates collected (0/5)
- [ ] Green plates collected (0/5)

Last updated: 2025-10-09
