"""FastAPI application: prediction, retrain, insights, and uptime monitoring."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CLASS_NAMES, DATA_DIR, MODEL_PATH, UPLOAD_DIR  # noqa: E402
from src.database import (  # noqa: E402
    count_uploads_by_label,
    get_uptime,
    init_db,
    latest_retrain_jobs,
    list_uploads,
    save_prediction,
    save_upload,
    touch_heartbeat,
)
from src.prediction import predict_from_bytes  # noqa: E402
from src.preprocessing import (  # noqa: E402
    dataset_counts,
    save_upload_image,
    summarize_image_stats,
)
from src.retrain import run_retrain  # noqa: E402

APP_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

app = FastAPI(
    title="Malaria MLOps",
    description="Image classification pipeline for malaria cell diagnosis",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")

STARTED_AT = time.time()
DEMO_STATS_PATH = DATA_DIR / "demo_stats.json"


def _demo_stats() -> dict:
    if DEMO_STATS_PATH.exists():
        return json.loads(DEMO_STATS_PATH.read_text())
    return {}


def _insight_payload() -> dict:
    train_counts = dataset_counts()
    stats = summarize_image_stats(sample_size=120)
    demo = _demo_stats()
    if train_counts.get("total", 0) == 0 and demo:
        train_counts = demo.get("train_counts", train_counts)
        stats = demo.get("image_stats", stats)
    return {
        "train_counts": train_counts,
        "upload_counts": count_uploads_by_label(),
        "image_stats": stats,
        "recent_uploads": list_uploads(20),
    }


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    touch_heartbeat()


@app.middleware("http")
async def heartbeat_middleware(request: Request, call_next):
    response = await call_next(request)
    try:
        touch_heartbeat()
    except Exception:
        pass
    return response


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    insights = _insight_payload()
    uptime = get_uptime()
    jobs = latest_retrain_jobs(5)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "class_names": CLASS_NAMES,
            "train_counts": insights["train_counts"],
            "upload_counts": insights["upload_counts"],
            "stats": insights["image_stats"],
            "uptime": uptime,
            "jobs": jobs,
            "model_ready": MODEL_PATH.exists(),
            "process_uptime": int(time.time() - STARTED_AT),
        },
    )


@app.get("/health")
async def health():
    uptime = get_uptime()
    return {
        "status": "ok",
        "model_loaded": MODEL_PATH.exists(),
        "uptime": uptime,
        "process_uptime_seconds": int(time.time() - STARTED_AT),
    }


@app.get("/api/insights")
async def insights():
    return _insight_payload()


@app.post("/api/predict")
async def api_predict(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        return JSONResponse({"error": "Empty file"}, status_code=400)
    try:
        result = predict_from_bytes(data)
    except Exception as exc:
        return JSONResponse({"error": f"Prediction failed: {exc}"}, status_code=400)

    save_prediction(file.filename or "upload", result["label"], result["confidence"])
    return result


@app.post("/api/upload")
async def api_upload(
    label: str = Form(...),
    files: list[UploadFile] = File(...),
):
    if label not in CLASS_NAMES:
        return JSONResponse(
            {"error": f"label must be one of {CLASS_NAMES}"}, status_code=400
        )
    saved = []
    for upload in files:
        data = await upload.read()
        if not data:
            continue
        try:
            path = save_upload_image(
                data, UPLOAD_DIR, upload.filename or "image.png", label
            )
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        save_upload(upload.filename or path.name, label, str(path))
        saved.append({"filename": path.name, "label": label, "path": str(path)})
    return {"saved": saved, "count": len(saved)}


@app.post("/api/retrain")
async def api_retrain(background_tasks: BackgroundTasks, sync: bool = False):
    """Trigger retraining on uploaded data. Use sync=true to wait for completion."""
    if sync:
        result = run_retrain()
        status = 200 if result.get("status") == "success" else 400
        return JSONResponse(result, status_code=status)

    background_tasks.add_task(run_retrain)
    return {
        "status": "started",
        "message": "Retraining started in the background. Refresh to see job status.",
    }


@app.get("/api/retrain/jobs")
async def retrain_jobs():
    return {"jobs": latest_retrain_jobs(20)}


@app.get("/api/uptime")
async def uptime():
    return get_uptime()
