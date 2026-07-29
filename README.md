# MalariaCell MLOps

End-to-end MLOps pipeline for malaria cell image classification using the NIH malaria cell images dataset.

**Stack:** FastAPI UI + API · SQLite · Docker · Locust · Render

---

## Training and deployment notes

| Stage | Recommended setup |
|-------|-------------------|
| Initial training | GPU if available (e.g. Google Colab), or CPU locally |
| Deploy / inference | Render free Docker web service |
| UI-triggered retrain | Head-only fine-tune on CPU (conv layers frozen) |

The production model is a compact custom CNN so inference and short retrains stay practical on free hosting. The saved model is reused as the starting point when retraining on new uploads.

---

## Repository structure

```text
Machine-Learning-OP/
├── README.md
├── notebook/
│   └── malaria_mlops.ipynb
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── prediction.py
├── data/
│   ├── train/
│   └── test/
└── models/
    └── malaria_cnn.keras
```

Additional files: `app/`, `Dockerfile`, `docker-compose.yml`, `locust/`, `render.yaml`, `scripts/`.

---

## Features

1. **Prediction** — upload one cell image → Parasitized / Uninfected + confidence  
2. **Upload data** — bulk images saved to disk and SQLite for retraining  
3. **Retrain trigger** — preprocess uploads → fine-tune the saved model  
4. **UI** — uptime monitoring, dataset insights, predict + retrain  
5. **API** — FastAPI (`/api/predict`, `/api/upload`, `/api/retrain`, `/health`)  
6. **Docker + Locust** — scale API containers and compare latency  

---

## Quick start (local)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1) Download data + create splits
python scripts/download_data.py --max-per-class 1500

# 2) Train and save models/malaria_cnn.keras
python scripts/train_initial.py --epochs 8

# 3) Run the app
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000

API docs: http://localhost:8000/docs

---

## Docker + Locust load testing

```bash
# Build and run with 1 API replica behind nginx
docker compose up --build -d --scale api=1
# App: http://localhost:8080

# Load test
docker compose --profile load up locust
# Locust UI: http://localhost:8089

# Scale to 3 API containers and compare latency
docker compose up -d --scale api=3
```


## Demo video

**Demo video:** [Malaria Cells prediction](https://youtu.be/P3OKJdE55Xo)


## API summary

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Uptime / readiness |
| GET | `/api/insights` | Dataset stats |
| POST | `/api/predict` | Multipart `file` image |
| POST | `/api/upload` | Form `label` + multipart `files` |
| POST | `/api/retrain?sync=true` | Blocking retrain |
| GET | `/api/retrain/jobs` | Job history |

---

## Dataset

NIH / NLM malaria cell images:  
https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip
