from fastapi import APIRouter

from app import store
from app.schemas import BoundariesRequest, PeekingRequest, SequentialAnalyzeRequest
from app.services import sequential as svc

router = APIRouter(prefix="/sequential", tags=["sequential"])


@router.post("/boundaries")
def boundaries(req: BoundariesRequest):
    looks = svc.resolve_looks(req.n_looks, req.looks)
    return svc.compute_boundaries(looks, req.alpha, req.spending)


@router.post("/analyze")
def analyze(req: SequentialAnalyzeRequest):
    looks = svc.resolve_looks(req.n_looks, req.looks)
    df = store.get(req.dataset_id)
    return svc.analyze_stream(df, req.group_col, req.outcome_col, req.control_label,
                              req.planned_total_n, looks, req.alpha, req.spending)


@router.post("/peeking-simulation")
def peeking(req: PeekingRequest):
    return svc.peeking_simulation(req.n_looks, req.alpha, req.n_sims, req.seed)
