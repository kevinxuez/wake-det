# wake-det — SAR Ship-Wake Surveillance Prototype

End-to-end pipeline for detecting ship wakes in SAR (Synthetic Aperture Radar)
imagery, correcting geolocation, and identifying dark (non-cooperative) vessels
via AIS correlation.

---

## Architecture Overview

```
┌──────────────────────────────────────────────┐
│  Step 1  Dataset Acquisition                 │
│  src/data/download_datasets.py               │
│  • Kaggle sar_wake (Gaofen-3)                │
│  • OpenSARWake (OBB annotations, 3 973 imgs) │
│  • 70 / 20 / 10 train/val/test split         │
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

### 2. Download datasets

```bash
# Configure Kaggle credentials first: https://github.com/Kaggle/kaggle-api#api-credentials
python src/data/download_datasets.py --output-dir data/
```

### 3. Generate dataset YAML and train

```bash
# Auto-generate YAML from downloaded splits
python src/train/train_yolov8_obb.py \
    --generate-dataset-yaml data/splits/sar_wake_kaggle

# Train (edit configs/train_config.yaml first to set GPU / batch size)
python src/train/train_yolov8_obb.py \
    --config configs/train_config.yaml \
    --data   data/splits/sar_wake_kaggle/dataset.yaml
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
├── src/
│   ├── data/
│   │   └── download_datasets.py   # Step 1: Dataset acquisition & splitting
│   ├── train/
│   │   └── train_yolov8_obb.py    # Step 2: YOLOv8-OBB training
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
| OpenSARWake | 3 973-image multi-band SAR wake dataset with OBB labels |
| Copernicus API | https://dataspace.copernicus.eu/analyse-and-process/apis |
| Azimuth shift | Raney (1971); Bamler & Hartl (1998) |
| AIS threshold | 150 m standard for cooperative-vessel matching |
