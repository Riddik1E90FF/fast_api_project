from contextlib import asynccontextmanager

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model_def import MoonClassifier


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = MoonClassifier()
    model.load_state_dict(torch.load("model/model.pth", map_location="cpu"))
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
