from fastapi import APIRouter

from app import store
from app.schemas import GroupSpec, MultipleTestingRequest
from app.services import corrections, frequentist
from app.services.groups import split_groups

router = APIRouter(prefix="/ab", tags=["ab-tests"])


def _arms(req: GroupSpec):
    c, t, cl, tl = split_groups(store.get(req.dataset_id), req.group_col, req.outcome_col, req.control_label)
    return c, t, cl, tl


@router.post("/t-test")
def t_test(req: GroupSpec):
    c, t, cl, tl = _arms(req)
    res = frequentist.welch_t_test(c, t, req.alpha)
    res.update(control_label=cl, treatment_label=tl)
    return res


@router.post("/chi-square")
def chi_square(req: GroupSpec):
    c, t, cl, tl = _arms(req)
    res = frequentist.chi_square_test(c, t, req.alpha)
    res.update(control_label=cl, treatment_label=tl)
    return res


@router.post("/multiple-testing")
def multiple_testing(req: MultipleTestingRequest):
    return corrections.correct(req.p_values, req.method, req.alpha, req.labels)
