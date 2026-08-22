"""
Train a vehicle detector/classifier from SCRATCH (random init, no pretrained
weights) on the Vehicles-coco.v2i.multiclass dataset.

Why random init and not model.pt?
The recruitment rules say pretrained models are not allowed unless stated.
Loading 'yolov8n.pt' would load COCO-pretrained weights. Loading
'yolov8n.yaml' builds the same architecture with random weights instead -
that's what makes this a "from scratch" submission while still using YOLO's
detection head/loss, which is fine since it's an architecture, not a
pretrained model.

Usage:
    python train.py --data "C:/Users/you/Downloads/Vehicles-coco.v2i.multiclass/data.yaml" --epochs 100

After training, best weights land in runs/detect/train/weights/best.pt
Copy that file to backend/app/weights/best.pt for the API to use it.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train vehicle detector from scratch")
    parser.add_argument("--data", required=True, help="Path to data.yaml from the Roboflow export")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960, help="Higher helps detect small/distant vehicles in top-down traffic-cam shots")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--arch",
        default="yolov8n.yaml",
        help="Architecture-only config (no pretrained weights). Use yolov8n/s/m.yaml",
    )
    parser.add_argument("--device", default=0, help="GPU index, or 'cpu'")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Could not find {data_path}. Point --data at the data.yaml "
            "inside your Vehicles-coco.v2i.multiclass folder."
        )

    # Build model from architecture yaml only -> random weights, not pretrained
    model = YOLO(args.arch)

    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        pretrained=False,   # belt-and-suspenders: never pull pretrained weights
        patience=20,
        project="runs/detect",
        name="train",
        exist_ok=True,
        # Stronger augmentation than YOLO's defaults - helps a scratch-trained
        # model see more visual variety than the raw dataset alone provides,
        # which matters a lot for generalizing beyond the dataset's own domain.
        degrees=10.0,       # random rotation
        translate=0.15,
        scale=0.6,
        shear=3.0,
        perspective=0.0005,
        hsv_h=0.02,         # hue jitter
        hsv_s=0.7,          # saturation jitter
        hsv_v=0.5,          # brightness jitter
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
    )

    metrics = model.val()
    print("\nValidation metrics:")
    print(metrics.results_dict)

    print(
        "\nDone. Copy runs/detect/train/weights/best.pt to "
        "backend/app/weights/best.pt to serve it from the API."
    )


if __name__ == "__main__":
    main()