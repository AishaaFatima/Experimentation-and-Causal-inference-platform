from fastapi import APIRouter

from app import store
from app.schemas import DIDRequest, DoWhyRequest, EconMLRequest, PSMRequest, PSMSensitivityRequest
from app.services import did, dowhy_analysis, econml_analysis, psm

router = APIRouter(prefix="/causal", tags=["causal"])


@router.post("/psm")
def propensity_matching(req: PSMRequest):
    return psm.psm_analysis(store.get(req.dataset_id), req.treat_col, req.outcome_col, req.covariates,
                            req.caliper_sd, req.n_neighbors, req.n_boot, req.alpha, req.seed,
                            req.squares, req.zero_flags)


@router.post("/psm-sensitivity")
def propensity_sensitivity(req: PSMSensitivityRequest):
    return psm.psm_sensitivity(store.get(req.dataset_id), req.treat_col, req.outcome_col, req.covariates,
                               req.caliper_sd, req.n_neighbors)


@router.post("/did")
def diff_in_diff(req: DIDRequest):
    return did.did_analysis(store.get(req.dataset_id), req.treat_col, req.pre_col, req.post_col,
                            req.covariates, req.placebo_pre_col, req.placebo_post_col, req.alpha)


@router.post("/dowhy")
def dowhy_estimate(req: DoWhyRequest):
    return dowhy_analysis.dowhy_analysis(store.get(req.dataset_id), req.treat_col, req.outcome_col, req.covariates,
                                         req.refute_with, req.n_simulations, req.alpha, req.seed)


@router.post("/econml")
def econml_estimate(req: EconMLRequest):
    return econml_analysis.econml_analysis(store.get(req.dataset_id), req.treat_col, req.outcome_col, req.covariates,
                                           req.effect_modifiers, req.n_trees, req.min_propensity, req.alpha, req.seed)
