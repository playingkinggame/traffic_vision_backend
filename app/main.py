import base64
import shutil
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from .detector import detector
from .tracker import mjpeg_track_stream, STATS_STORE

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

JOBS: dict[str, Path] = {}  # job_id -> uploaded video path, for /api/track/*

app = FastAPI(title="AI Traffic Police - Vehicle Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your frontend origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_ready": detector.is_ready,
        "classes": list(detector.class_names.values()) if detector.is_ready else [],
    }


@app.post("/api/predict/image")
async def predict_image(file: UploadFile = File(...), conf: float = 0.4):
    if not detector.is_ready:
        raise HTTPException(503, "Model not trained yet. Run train.py and place best.pt in app/weights/.")

    raw = await file.read()
    npimg = np.frombuffer(raw, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Could not decode image.")

    annotated, detections, counts = detector.predict_image(image, conf=conf)

    ok, buf = cv2.imencode(".jpg", annotated)
    annotated_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

    return {
        "detections": detections,
        "counts": counts,
        "total_vehicles": sum(counts.values()),
        "annotated_image_base64": f"data:image/jpeg;base64,{annotated_b64}",
    }


@app.post("/api/predict/video")
async def predict_video(file: UploadFile = File(...), sample_every: int = 1):
    if not detector.is_ready:
        raise HTTPException(503, "Model not trained yet. Run train.py and place best.pt in app/weights/.")

    job_id = uuid.uuid4().hex[:10]
    in_path = OUTPUT_DIR / f"{job_id}_in.mp4"
    out_path = OUTPUT_DIR / f"{job_id}_out.mp4"

    with in_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        summary = detector.predict_video(str(in_path), str(out_path), sample_every=sample_every)
    finally:
        in_path.unlink(missing_ok=True)

    summary["annotated_video_url"] = f"/outputs/{out_path.name}"
    return summary


@app.get("/")
def root():
    return {
        "message": "AI Traffic Police backend running.",
        "endpoints": [
            "/api/health",
            "/api/predict/image",
            "/api/predict/video",
            "/api/track/upload",
            "/api/track/stream/{job_id}",
            "/api/track/stats/{job_id}",
        ],
    }


# ---------------------------------------------------------------------------
# Live tracking: upload once, then stream annotated frames as they're
# processed (MJPEG), while stats update in near-real-time on a side panel.
# ---------------------------------------------------------------------------

@app.post("/api/track/upload")
async def track_upload(file: UploadFile = File(...)):
    if not detector.is_ready:
        raise HTTPException(503, "Model not trained yet. Run train.py and place best.pt in app/weights/.")

    job_id = uuid.uuid4().hex[:10]
    in_path = OUTPUT_DIR / f"{job_id}_track_in.mp4"
    with in_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    JOBS[job_id] = in_path
    return {"job_id": job_id}


@app.get("/api/track/stream/{job_id}")
def track_stream(job_id: str, conf: float = 0.4, mpp: float = 0.05):
    in_path = JOBS.get(job_id)
    if in_path is None or not in_path.exists():
        raise HTTPException(404, "Unknown or expired job_id. Upload the video again.")

    return StreamingResponse(
        mjpeg_track_stream(str(in_path), job_id, conf=conf, mpp=mpp),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/track/stats/{job_id}")
def track_stats(job_id: str):
    return STATS_STORE.get(job_id, {"unique_counts": {}, "total_unique": 0, "current": [], "done": False})