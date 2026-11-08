from fastapi import APIRouter

from app import store
from app.schemas import BayesABRequest
from app.services import bayes

router = APIRouter(prefix="/bayes", tags=["bayesian"])


@router.post("/ab")
def bayesian_ab(req: BayesABRequest):
    return bayes.bayesian_ab(store.get(req.dataset_id), req.group_col, req.outcome_col, req.control_label,
                             req.metric, req.prior_alpha, req.prior_beta, req.draws, req.tune, req.chains,
                             req.min_effect, req.seed, req.likelihood)
