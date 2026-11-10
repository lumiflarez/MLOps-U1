import os
from contextlib import asynccontextmanager

import mlflow
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient
from pydantic import BaseModel, Field

MODEL_NAME = os.environ.get("MODEL_NAME", "jailbreak-detector")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "champion")
LABELS = {0: "benign", 1: "jailbreak"}

state = {"model": None, "version": None, "run_id": None, "error": None}

def load_champion() -> None:
    client = MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    state["model"] = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{mv.version}")
    state["version"] = int(mv.version)
    state["run_id"] = mv.run_id

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_champion()
    except Exception as exc:  # noqa: BLE001 - bet kokia klaida -> /health 503
        state["error"] = str(exc)
    yield

app = FastAPI(title="Jailbreak prompt detector", lifespan=lifespan)

class PredictRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=10000)

@app.get("/health")
def health():
    if state["model"] is None:
        raise HTTPException(status_code=503,
                            detail={"status": "no model", "error": state["error"]})
    return {"status": "ok",
            "source": f"models:/{MODEL_NAME}@{MODEL_ALIAS}",
            "version": state["version"],
            "run_id": state["run_id"]}

@app.post("/predict")
def predict(req: PredictRequest):
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    model = state["model"]
    pred = int(model.predict([req.prompt])[0])
    proba = float(model.predict_proba([req.prompt])[0][1])
    return {"prediction": pred,
            "label": LABELS[pred],
            "jailbreak_probability": round(proba, 4),
            "model_version": state["version"]}