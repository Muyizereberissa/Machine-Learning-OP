"""Locust load test for prediction health + predict endpoints."""

from __future__ import annotations

import io
import random

from locust import HttpUser, between, task
from PIL import Image


def _fake_cell_bytes() -> bytes:
    """Generate a tiny RGB PNG so load tests do not need real dataset files."""
    color = (
        random.randint(40, 200),
        random.randint(20, 120),
        random.randint(40, 180),
    )
    img = Image.new("RGB", (128, 128), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class MalariaUser(HttpUser):
    wait_time = between(0.2, 1.0)

    @task(3)
    def health(self):
        self.client.get("/health")

    @task(2)
    def insights(self):
        self.client.get("/api/insights")

    @task(5)
    def predict(self):
        files = {"file": ("cell.png", _fake_cell_bytes(), "image/png")}
        self.client.post("/api/predict", files=files)
