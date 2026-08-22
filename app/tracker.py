"""
Adds persistent vehicle tracking (ByteTrack, bundled with ultralytics),
per-vehicle speed estimation from centroid displacement, and a live MJPEG
frame generator so the frontend can watch detection happen as the video
plays, instead of waiting for a fully-processed file.

Speed estimation caveat (be upfront about this in your README): true
real-world speed needs the camera's meters-per-pixel ratio, which depends
on camera height/angle and isn't knowable from the video alone. `mpp`
below is a rough default - calibrate it properly by measuring a known
real-world distance in your footage (e.g. a lane's painted width, ~3m)
against how many pixels it spans, then set mpp = real_meters / pixel_span.
Until calibrated, treat the km/h numbers as indicative, not exact.
"""
import math
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2

from .detector import detector, CLASS_COLORS

# job_id -> live stats dict, polled by the frontend while streaming runs
STATS_STORE: dict[str, dict] = {}


def _color_for(cls_id: int):
    return CLASS_COLORS[cls_id % len(CLASS_COLORS)]


def mjpeg_track_stream(video_path: str, job_id: str, conf: float = 0.4, mpp: float = 0.05):
    """Generator yielding multipart JPEG chunks - point an <img> tag's src
    at the endpoint using this, and the browser renders it as a live feed.
    """
    if not detector.is_ready:
        raise RuntimeError("Model weights not found. Train the model first.")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = 1.0 / fps

    # track_id -> deque of (frame_idx, cx, cy), just enough history for speed calc
    track_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=5))
    seen_ids: dict[str, set] = defaultdict(set)  # class label -> set of track ids ever seen
    frame_idx = 0

    STATS_STORE[job_id] = {
        "unique_counts": {},
        "total_unique": 0,
        "current": [],
        "frame": 0,
        "done": False,
    }

    while True:
        t_start = time.time()
        ok, frame = cap.read()
        if not ok:
            break

        results = detector.model.track(
            frame, persist=True, tracker="bytetrack.yaml", conf=conf, verbose=False
        )[0]

        annotated = frame.copy()
        current_frame_stats = []

        if results.boxes.id is not None:
            ids = results.boxes.id.int().cpu().tolist()
            cls_ids = results.boxes.cls.int().cpu().tolist()
            xyxy = results.boxes.xyxy.cpu().tolist()

            for box, track_id, cls_id in zip(xyxy, ids, cls_ids):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                label = detector.class_names.get(cls_id, str(cls_id))
                seen_ids[label].add(track_id)

                speed_kmh = None
                hist = track_history[track_id]
                if hist:
                    prev_frame_idx, prev_cx, prev_cy = hist[-1]
                    dt = (frame_idx - prev_frame_idx) / fps
                    if dt > 0:
                        dist_px = math.hypot(cx - prev_cx, cy - prev_cy)
                        speed_mps = (dist_px * mpp) / dt
                        speed_kmh = round(speed_mps * 3.6, 1)
                hist.append((frame_idx, cx, cy))

                current_frame_stats.append(
                    {"id": track_id, "label": label, "speed_kmh": speed_kmh}
                )

                color = _color_for(cls_id)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                tag = f"#{track_id} {label}"
                if speed_kmh is not None:
                    tag += f" {speed_kmh:.0f}km/h"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
                cv2.putText(
                    annotated, tag, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 1, cv2.LINE_AA,
                )

        STATS_STORE[job_id] = {
            "unique_counts": {k: len(v) for k, v in seen_ids.items()},
            "total_unique": sum(len(v) for v in seen_ids.values()),
            "current": current_frame_stats,
            "frame": frame_idx,
            "done": False,
        }

        ok2, buf = cv2.imencode(".jpg", annotated)
        if ok2:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            )

        frame_idx += 1
        # pace to source fps when we're processing faster than real-time;
        # if inference is slower than fps allows, just send the next frame
        # as soon as it's ready (still "live", just at whatever pace the
        # hardware can sustain - same trade-off any live CV demo makes on CPU)
        elapsed = time.time() - t_start
        remaining = frame_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)

    cap.release()
    if job_id in STATS_STORE:
        STATS_STORE[job_id]["done"] = True
        STATS_STORE[job_id]["current"] = []
    Path(video_path).unlink(missing_ok=True)
