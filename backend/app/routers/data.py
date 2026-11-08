import io
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from app import store
from app.schemas import DatasetInfo, SimulateRequest
from app.services.simulator import simulate_ab

router = APIRouter(prefix="/data", tags=["data"])
MAX_BYTES = 20 * 1024 * 1024
LALONDE_PATH = Path(__file__).resolve().parents[2] / "data" / "lalonde.csv"


@router.post("/upload", response_model=DatasetInfo)
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "File too large (20 MB max)")
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(422, f"Could not parse CSV: {e}")
    if df.empty:
        raise HTTPException(422, "CSV has no rows")
    return store.register(df, file.filename or "upload.csv")


@router.post("/sample/lalonde", response_model=DatasetInfo)
def load_lalonde():
    df = pd.read_csv(LALONDE_PATH)
    notes = ("LaLonde / Dehejia-Wahba job-training data (observational version): 185 program participants "
             "(treat=1) compared with 429 people from the PSID survey (treat=0). Outcome: re78 (1978 earnings). "
             "Assignment was NOT random here, so a naive comparison is biased.")
    return store.register(df, "lalonde.csv", notes=notes)


@router.post("/simulate", response_model=DatasetInfo)
def simulate(req: SimulateRequest):
    df = simulate_ab(req.n_per_group, req.metric, req.baseline_rate, req.lift, req.mean, req.std, req.seed)
    return store.register(df, "simulated_ab", notes="Simulated A/B data. Groups are 'A' (control) and 'B'.",
                          true_effect=req.lift)


@router.get("/{dataset_id}", response_model=DatasetInfo)
def get_dataset(dataset_id: str):
    df = store.get(dataset_id)
    info = store.register(df, "copy")  # cheap way to rebuild the description
    store._STORE.pop(info.dataset_id)
    info.dataset_id = dataset_id
    return info
