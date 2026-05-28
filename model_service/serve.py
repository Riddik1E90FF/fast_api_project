import os
from contextlib import asynccontextmanager

import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model_def import MoonClassifier

MODEL_PATH = "model/model.pth"


def _train_and_save():
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, TensorDataset

    print("model.pth not found — training now...")
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=32, shuffle=True)

    m = MoonClassifier()
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    for _ in range(100):
        for xb, yb in loader:
            opt.zero_grad()
            criterion(m(xb), yb).backward()
            opt.step()

    os.makedirs("model", exist_ok=True)
    torch.save(m.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return m


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(MODEL_PATH):
        model = _train_and_save()
    else:
        model = MoonClassifier()
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    app.state.model = model
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": app.state.model is not None}


@app.post("/predict")
def predict(req: PredictionRequest):
    if len(req.features) != 2:
        raise HTTPException(status_code=422, detail="Expected exactly 2 features [x1, x2]")

    tensor = torch.tensor([req.features], dtype=torch.float32)

    with torch.no_grad():
        logits = app.state.model(tensor)
        probs = torch.softmax(logits, dim=1)
        pred_class = probs.argmax(dim=1).item()
        confidence = probs[0][pred_class].item()

    return {
        "prediction": f"class_{pred_class}",
        "confidence": round(confidence, 4),
        "model": "moon-classifier-v1",
    }
