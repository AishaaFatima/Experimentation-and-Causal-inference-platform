from fastapi import APIRouter

from app.schemas import PowerCurveRequest, PowerRequest, SampleSizeRequest
from app.services import power as svc

router = APIRouter(prefix="/power", tags=["power"])


def _common(r):
    return dict(metric=r.metric, mde=r.mde, alpha=r.alpha, ratio=r.ratio, alternative=r.alternative,
                baseline_rate=r.baseline_rate, std=r.std)


@router.post("/sample-size")
def sample_size(req: SampleSizeRequest):
    res = svc.sample_size(power=req.power, **_common(req))
    res["summary"] = (f"You need about {res['n_control']:,} users in control and {res['n_treatment']:,} in "
                      f"treatment ({res['n_total']:,} total) to detect a change of {req.mde:+g} with "
                      f"{req.power:.0%} power at alpha = {req.alpha}.")
    return res


@router.post("/achieved-power")
def achieved_power(req: PowerRequest):
    p = svc.achieved_power(n_per_group=req.n_per_group, **_common(req))
    return {"power": p, "summary": f"With {req.n_per_group:,} per group you have {p:.0%} power to detect a change of {req.mde:+g}."}


@router.post("/curve")
def curve(req: PowerCurveRequest):
    return svc.power_curve(power=req.power, **_common(req))
