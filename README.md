# Backend — Vehicle Detection API

FastAPI service that wraps a YOLOv8 model (trained from scratch — no
pretrained weights, per the challenge rules) and exposes it for the
frontend to call.

## Setup

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

## 1. Convert the dataset (one-time)

The provided dataset is a **COCO-format** Roboflow export (note the
`-coco` in the folder name) — it has `_annotations.coco.json` per split
and no `data.yaml`, so `ultralytics` can't read it directly. Convert it
to YOLO layout first:

```bash
python convert_coco_to_yolo.py --src "C:/Users/yathin/Downloads/Vehicles-coco.v2i.multiclass" --out "C:/Users/yathin/Downloads/vehicles-yolo"
```

This copies images into `images/` + writes YOLO `.txt` labels into
`labels/` for each split, and generates `data.yaml`. It also prints the
class index mapping it found — check it lines up with car/bus/bike/truck.

(Alternative: re-export the dataset from Roboflow choosing "YOLOv8"
format instead of "COCO" — that gives you `data.yaml` directly and you
can skip this step.)

## 2. Train the model

Point `--data` at the `data.yaml` the conversion script just created:

```bash
python train.py --data "C:/Users/yathin/Downloads/vehicles-yolo/data.yaml" --epochs 100
```

This builds a YOLOv8n architecture with **random initial weights**
(`yolov8n.yaml`, not `yolov8n.pt`), so nothing pretrained is used — the
model learns car/bus/bike/truck purely from your dataset.

On CPU this will be slow; a free Colab/Kaggle GPU runtime is worth using —
just run the same script there and download `best.pt` afterwards.

When training finishes:

```bash
copy runs\detect\train\weights\best.pt app\weights\best.pt
```

## 3. Run the API

```bash
uvicorn app.main:app --reload --port 8000
```

Check it's alive: http://localhost:8000/api/health — `model_ready` should
be `true` once `best.pt` is in place.

## Endpoints

| Method | Path                  | Purpose                                  |
|--------|-----------------------|-------------------------------------------|
| GET    | `/api/health`         | Status + class names                      |
| POST   | `/api/predict/image`  | Upload an image, get boxes + counts       |
| POST   | `/api/predict/video`  | Upload a video, get an annotated copy + timeline of counts |

`predict/video` accepts an optional `?sample_every=N` query param to run
inference every Nth frame (useful on CPU — reuses the last frame's boxes
in between).
