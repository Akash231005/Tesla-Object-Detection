# Tesla Autopilot Clone: Object Detection

Road-object detection inspired by autonomous-driving perception, built with **YOLOv8 (Ultralytics)** on the **KITTI** object-detection dataset. The whole project is a single, Kaggle-ready Jupyter notebook that goes from raw KITTI files to a trained, evaluated and saved detector.

> Educational project. It is *inspired by* driver-assistance perception and is not affiliated with Tesla and does not reproduce Tesla's Autopilot stack.

## Use case

Better object detection and tracking is a building block for safer, more efficient autonomous driving. This project develops a computer-vision model that detects vehicles, pedestrians and other road users in front-camera images.

## Detected classes

| ID | Class |
|----|-------|
| 0 | Car |
| 1 | Pedestrian |
| 2 | Van |
| 3 | Cyclist |
| 4 | Truck |
| 5 | Misc |
| 6 | Tram |
| 7 | Person_sitting |

KITTI's `DontCare` regions mark areas that were *not* labelled. They are not an object class, so they are excluded from the YOLO labels.

## What the notebook does

1. **Auto-detects the KITTI dataset** under `/kaggle/input/` (no hard-coded folder names) and prints a clear diagnostic if it is missing.
2. **Explores the data**: directory tree, image sizes, class distribution, objects per image, box-size distribution, annotated samples.
3. **Converts KITTI → YOLO labels** (pixel `x1,y1,x2,y2` → normalised `class xc yc w h`) with clipping and invalid-box handling.
4. **Verifies the conversion** with format checks, a numerical round-trip and visual overlays.
5. **Creates a reproducible split** (default 80/10/10, seed 42) and a validated `data.yaml`.
6. **Fine-tunes YOLOv8n** (configurable model, image size, epochs, batch size, patience).
7. **Evaluates `best.pt`** on validation *and* test splits: precision, recall, mAP@0.5, mAP@0.5:0.95, per-class table, confusion matrix.
8. **Visualises predictions** on a random test sample, provides a reusable `predict_image()` function, and saves the model and metrics.

All reported numbers are produced by running the notebook; nothing is hard-coded.

## Repository contents

```
.
├── Tesla_Autopilot_Clone_Object_Detection_KITTI_YOLOv8.ipynb   # the project
├── requirements.txt                                            # dependencies (for local use)
└── README.md
```

## Run on Kaggle (recommended)

1. Create a new Kaggle Notebook and import the `.ipynb` file (File → Import Notebook).
2. Turn on a GPU: *Session options → Accelerator → GPU T4 x2 / P100*.
3. Turn **Internet On** (needed for installing `ultralytics` and downloading the YOLOv8 weights).
4. Click **Add Input**, search for "kitti" and add a KITTI object-detection dataset, e.g. [`klemenko/kitti-dataset`](https://www.kaggle.com/datasets/klemenko/kitti-dataset) (contains `data_object_image_2` and `data_object_label_2`).
5. Click **Run All**.

Expected input layout (other layouts work as long as images and KITTI `.txt` labels share file names):

```
/kaggle/input/<dataset-name>/
    data_object_image_2/training/image_2/000000.png ...
    data_object_label_2/training/label_2/000000.txt ...
```

Generated files go to `/kaggle/working/`:

```
/kaggle/working/
├── kitti_yolo/             # generated YOLO dataset (images/, labels/, data.yaml)
├── runs/                   # Ultralytics training + validation outputs
└── final_outputs/          # best.pt, data.yaml, class_mapping.json, metrics, figures
```

## Configuration

All settings are in **Section 4** of the notebook:

| Variable | Default | Meaning |
|----------|---------|---------|
| `MODEL_NAME` | `yolov8n.pt` | Try `yolov8s.pt` / `yolov8m.pt` for higher accuracy |
| `IMG_SIZE` | `640` | Training/inference size (must be a multiple of 32) |
| `EPOCHS` | `30` | Training epochs |
| `BATCH_SIZE` | `16` | Lower to 8 if you hit CUDA out-of-memory |
| `PATIENCE` | `10` | Early-stopping patience |
| `CONF_THRESHOLD` | `0.25` | Confidence threshold for visualised predictions |
| `TRAIN_FRAC / VAL_FRAC / TEST_FRAC` | `0.8 / 0.1 / 0.1` | Split ratios |
| `MAX_IMAGES` | `None` | Set e.g. `300` for a quick smoke test |
| `DELETE_DATASET_COPY_AT_END` | `False` | `True` frees ~6 GB of output space when the run ends |

## Using the trained model

```python
from ultralytics import YOLO

model = YOLO("final_outputs/best.pt")
results = model.predict("some_road_image.jpg", conf=0.25)
results[0].show()
```

## Running locally

The notebook paths are Kaggle-specific (`/kaggle/input`, `/kaggle/working`). To run elsewhere, install `requirements.txt`, download the KITTI object-detection images and `label_2` labels, and change the path variables in Section 4. A CUDA GPU is strongly recommended; CPU training is very slow.

```bash
pip install -r requirements.txt
```

## Known limitations

- **Possible split leakage:** KITTI frames come from continuous drives and the object-detection split has no sequence IDs, so a random split can put near-identical neighbouring frames in different splits. Scores may be optimistic compared with a sequence-level split.
- **`DontCare` is discarded, not masked:** detections in unlabelled regions count as false positives.
- Single lightweight model, single run, no hyper-parameter search; images are down-scaled to `IMG_SIZE`, which penalises small or distant objects.
- Detection only: no tracking, depth estimation or sensor fusion.

## Ideas for improvement

- Larger models (`yolov8s/m`), larger `IMG_SIZE` (e.g. 960), more epochs.
- Sequence-aware splitting; class balancing for rare classes (`Person_sitting`, `Misc`).
- Mask `DontCare` regions during evaluation.
- Add multi-object tracking across video frames for the "detection and tracking" use case.

## Data, licences and acknowledgements

- **KITTI Vision Benchmark Suite** (Karlsruhe Institute of Technology & Toyota Technological Institute at Chicago). Check the [official KITTI terms](https://www.cvlibs.net/datasets/kitti/) before use; the data is generally restricted to non-commercial/academic use. The Kaggle mirror lists its licence as "Other".
- **Ultralytics YOLOv8** is released under AGPL-3.0 (with an enterprise licence available); check its terms for your use case.
- Project structure initially based on a GUVI final-project outline ("Tesla Autopilot Clone: Object Detection").

Add your own `LICENSE` file for the notebook code if you publish it.
