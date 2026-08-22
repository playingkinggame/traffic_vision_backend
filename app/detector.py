"""
Wraps the trained YOLO model: loading, running inference, and drawing
HUD-style annotations (matches the project's teal/purple/pink look) on
images and video frames.
"""
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
from ultralytics import YOLO

WEIGHTS_PATH = Path(__file__).parent / "weights" / "best.pt"

# BGR (OpenCV order) - teal / purple / pink / amber, one per class fallback
CLASS_COLORS = [
    (212, 175, 55),   # amber-ish
    (200, 120, 255),  # pink
    (200, 90, 130),   # purple
    (210, 200, 40),   # teal
]


class VehicleDetector:
    def __init__(self, weights_path: Path = WEIGHTS_PATH):
        self.weights_path = Path(weights_path)
        self.model = None
        self.class_names = {}
        self._load()

    def _load(self):
        if not self.weights_path.exists():
            # No trained weights yet - API still boots so the frontend can
            # show a clear "model not trained" state instead of crashing.
            self.model = None
            return
        self.model = YOLO(str(self.weights_path))
        self.class_names = self.model.names

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def _color_for(self, cls_id: int):
        return CLASS_COLORS[cls_id % len(CLASS_COLORS)]

    def predict_image(self, image: np.ndarray, conf: float = 0.25, imgsz: int = 960):
        """Run detection on a single BGR image (numpy array).

        Returns (annotated_image_bgr, detections, counts)
        """
        if not self.is_ready:
            raise RuntimeError("Model weights not found. Train the model first.")

        results = self.model.predict(image, conf=conf, imgsz=imgsz, verbose=False)[0]
        annotated = image.copy()
        detections = []
        counts = Counter()

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = self.class_names.get(cls_id, str(cls_id))
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            counts[label] += 1
            detections.append(
                {
                    "label": label,
                    "confidence": round(confidence, 3),
                    "box": [x1, y1, x2, y2],
                }
            )

            color = self._color_for(cls_id)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            text = f"{label} {confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(
                annotated, text, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 1, cv2.LINE_AA,
            )

        return annotated, detections, dict(counts)

    def predict_video(self, in_path: str, out_path: str, conf: float = 0.25, sample_every: int = 1):
        """Run detection across a video, writing an annotated copy.

        sample_every: run inference every Nth frame (>1 speeds things up on CPU),
        reusing the last result's boxes for skipped frames.

        Returns aggregate counts (max concurrent per class, seen-total per class)
        and per-second timeline of counts for charting.
        """
        if not self.is_ready:
            raise RuntimeError("Model weights not found. Train the model first.")

        cap = cv2.VideoCapture(in_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        frame_idx = 0
        last_counts = {}
        max_counts = Counter()
        timeline = []  # [{t: seconds, counts: {...}}]
        seen_ids_per_class = Counter()  # rough total-seen approx (per-frame sum, not tracked IDs)

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % sample_every == 0:
                annotated, _, counts = self.predict_image(frame, conf=conf)
                last_counts = counts
                for k, v in counts.items():
                    max_counts[k] = max(max_counts[k], v)
                    seen_ids_per_class[k] += v
            else:
                annotated = frame  # reuse raw frame between sampled ones (cheap)

            writer.write(annotated)

            if frame_idx % max(1, int(fps)) == 0:  # once per second
                timeline.append({"t": round(frame_idx / fps, 1), "counts": last_counts})

            frame_idx += 1

        cap.release()
        writer.release()

        return {
            "frames_processed": frame_idx,
            "duration_sec": round(frame_idx / fps, 1),
            "peak_concurrent_counts": dict(max_counts),
            "timeline": timeline,
        }


detector = VehicleDetector()