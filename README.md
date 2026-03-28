# wake-det — SAR Ship-Wake Surveillance Prototype

End-to-end pipeline for detecting ship wakes in SAR (Synthetic Aperture Radar)
imagery, correcting geolocation, and identifying dark (non-cooperative) vessels
via AIS correlation.

---

## Architecture Overview

```
┌──────────────────────────────────────────────┐
│  Step 1  OpenSARWake Dataset                 │
│  data/splits/{train,val,test}/{images,labels}│
│  • 3 973 images (L/C/X-band SAR, OBB labels) │
│  • 1024×1024 px, pre-split 60/20/20          │
└───────────────────┬──────────────────────────┘
                    │
┌───────────────────▼──────────────────────────┐
│  Step 2  Model Training                      │
│  src/train/train_yolov8_obb.py               │
│  • YOLOv8-OBB (anchor-free, rotation-aware)  │
│  • 1024×1024 input, ~86 % C-band accuracy    │
└───────────────────┬──────────────────────────┘
                    │
┌───────────────────▼──────────────────────────┐
│  Step 3  Inference Pipeline                  │
│  src/inference/inference_pipeline.py         │
│  • Overlapping tile strategy (default 1024 px│
│    tiles, 128 px overlap)                    │
│  • Oriented-bounding-box NMS                 │
└───────────────────┬──────────────────────────┘
                    │
┌───────────────────▼──────────────────────────┐
│  Step 4  SAR Data API                        │
│  src/api/sar_data_api.py                     │
│  • CopernicusClient  (Sentinel-1 GRD, free)  │
│  • ICEYEClient       (commercial)            │
│  • CapellaClient     (commercial)            │
└───────────────────┬──────────────────────────┘
                    │
┌───────────────────▼──────────────────────────┐
│  Step 5  Geolocation Correction              │
│  src/processing/geolocation.py               │
│  • Azimuth shift:  ζ = (R₀ / V) · v_r       │
│  • Anchors hull to wake apex                 │
│  • Outputs true geographic coordinates       │
└───────────────────┬──────────────────────────┘
                    │
        ┌───────────┴────────────┐
        ▼                        ▼
┌───────────────┐    ┌──────────────────────────┐
│  Step 6a      │    │  Step 6b                 │
│  Platform     │    │  AIS Correlation         │
│  Integration  │    │  src/integration/        │
│  src/         │    │  ais_correlation.py      │
│  integration/ │    │  • Interpolate AIS to    │
│  platform_    │    │    SAR acquisition time  │
│  integration  │    │  • Flag dark vessels     │
│  .py          │    │    (no AIS within 150 m) │
└───────────────┘    └──────────────────────────┘
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the OpenSARWake dataset

Place the pre-split OpenSARWake data under `data/splits/`:

```
data/splits/
├── train/
│   ├── images/   (2 383 × 1024×1024 .png)
│   └── labels/   (YOLO OBB .txt)
├── val/
│   ├── images/   (794 images)
│   └── labels/
└── test/
    ├── images/   (796 images)
    └── labels/
```

### 3. Train (choose one)

**Option A — YOLOv8-OBB** (fast inference, anchor-free):

```bash
python src/train/train_yolov8_obb.py \
    --generate-dataset-yaml data/splits

python src/train/train_yolov8_obb.py \
    --config configs/train_config.yaml \
    --data   data/splits/dataset.yaml
```

**Option B — Oriented R-CNN** (higher mAP, two-stage, per OpenSARWake paper):

```bash
# Requires: pip install mmengine mmcv mmdet mmrotate
python src/train/train_oriented_rcnn.py \
    --config configs/train_config.yaml \
    --data-root data/splits
```

### 4. Run inference on a SAR scene

```bash
python src/inference/inference_pipeline.py \
    --weights  runs/train/weights/best.pt \
    --image    data/scene.tif \
    --output   results/detections.json \
    --tile-size 1024 \
    --overlap   128
```

### 5. Apply geolocation correction

```python
from src.processing.geolocation import apply_geolocation_correction
import json

with open("results/detections.json") as f:
    detections = json.load(f)

# transform from GeoTIFF metadata (rasterio: dataset.transform)
transform = (120.0, 8.983e-5, 0.0, 35.0, 0.0, -8.983e-5)

corrected = apply_geolocation_correction(
    detections,
    transform=transform,
    slant_range=700_000,       # metres
    satellite_velocity=7_500,  # m/s (Sentinel-1)
)
```

### 6. Correlate with AIS to find dark vessels

```python
from src.integration.ais_correlation import correlate_detections

# ais_tracks: dict mapping MMSI → list of {timestamp, lon, lat} pings
dark_vessel_detections = correlate_detections(
    detections=corrected,
    ais_tracks=ais_tracks,
    acquisition_time="2024-06-15T10:32:00Z",
    threshold_m=150.0,
)

for d in dark_vessel_detections:
    if d["dark_vessel"]:
        print(f"DARK VESSEL at ({d['lon']:.5f}, {d['lat']:.5f})")
```

### 7. Push to intelligence platform

```python
from src.integration.platform_integration import PlatformClient

client = PlatformClient(
    base_url="https://platform.example.com/api",
    api_key="your_platform_api_key",
)

# Ingest detections
client.push_detections(dark_vessel_detections)

# Stream via webhook
client.send_webhook(
    dark_vessel_detections,
    webhook_url="https://hooks.example.com/sar",
)
```

---

## Project Structure

```
wake-det/
├── configs/
│   └── train_config.yaml          # Training hyper-parameters
├── data/
│   └── splits/
│       ├── train/{images,labels}/  # OpenSARWake training split
│       ├── val/{images,labels}/    # OpenSARWake validation split
│       └── test/{images,labels}/   # OpenSARWake test split
├── src/
│   ├── train/
│   │   ├── train_yolov8_obb.py    # Step 2a: YOLOv8-OBB training
│   │   └── train_oriented_rcnn.py # Step 2b: Oriented R-CNN (paper)
│   ├── inference/
│   │   └── inference_pipeline.py  # Step 3: Tiled inference + NMS
│   ├── api/
│   │   └── sar_data_api.py        # Step 4: Copernicus / ICEYE / Capella
│   ├── processing/
│   │   └── geolocation.py         # Step 5: Azimuth-shift correction
│   └── integration/
│       ├── platform_integration.py # Step 6a: REST / webhook ingestion
│       └── ais_correlation.py      # Step 6b: Dark vessel identification
├── tests/
│   ├── test_inference_pipeline.py
│   ├── test_geolocation.py
│   └── test_ais_correlation.py
├── requirements.txt
└── README.md
```

---

## Running Tests

```bash
pip install pytest numpy shapely
pytest tests/ -v
```

---

## Key References

| Component | Reference |
|-----------|-----------|
| YOLOv8-OBB | Ultralytics YOLO (https://docs.ultralytics.com/tasks/obb/) |
| OpenSARWake | 3 973-image multi-band SAR wake dataset (L/C/X-band, 1024×1024, OBB labels) |
| Copernicus API | https://dataspace.copernicus.eu/analyse-and-process/apis |
| Azimuth shift | Raney (1971); Bamler & Hartl (1998) |
| AIS threshold | 150 m standard for cooperative-vessel matching |
