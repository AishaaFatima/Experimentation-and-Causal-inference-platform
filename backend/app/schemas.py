"""Pydantic request/response models shared by the routers."""
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field, model_validator

Metric = Literal["binary", "continuous"]
Alternative = Literal["two-sided", "larger"]
Spending = Literal["obrien_fleming", "pocock"]
Prob = Annotated[float, Field(ge=0, le=1)]


# ---------- datasets ----------
class ColumnInfo(BaseModel):
    name: str
    dtype: str
    n_unique: int
    n_missing: int


class DatasetInfo(BaseModel):
    dataset_id: str
    name: str
    n_rows: int
    columns: list[ColumnInfo]
    preview: list[dict]
    notes: Optional[str] = None
    true_effect: Optional[float] = None  # only set for simulated data


class SimulateRequest(BaseModel):
    n_per_group: int = Field(1000, ge=10, le=200_000)
    metric: Metric = "binary"
    baseline_rate: float = Field(0.10, gt=0, lt=1)
    mean: float = 100.0
    std: float = Field(20.0, gt=0)
    lift: float = Field(0.02, description="Absolute true effect (B minus A)")
    seed: Optional[int] = None


class GroupSpec(BaseModel):
    dataset_id: str
    group_col: str
    outcome_col: str
    control_label: Optional[str] = None  # default: first level in sorted order
    alpha: float = Field(0.05, gt=0, lt=1)


# ---------- power ----------
class EffectSpec(BaseModel):
    metric: Metric = "binary"
    baseline_rate: Optional[float] = Field(None, gt=0, lt=1, description="Needed for binary metrics")
    std: Optional[float] = Field(None, gt=0, description="Needed for continuous metrics")
    mde: float = Field(..., description="Minimum detectable effect, absolute (e.g. 0.02 = +2 points)")
    alpha: float = Field(0.05, gt=0, lt=1)
    ratio: float = Field(1.0, gt=0, description="Treatment size / control size")
    alternative: Alternative = "two-sided"

    @model_validator(mode="after")
    def _check(self):
        if self.metric == "binary" and self.baseline_rate is None:
            raise ValueError("baseline_rate is required for binary metrics")
        if self.metric == "continuous" and self.std is None:
            raise ValueError("std is required for continuous metrics")
        if self.mde == 0:
            raise ValueError("mde must be non-zero")
        return self


class SampleSizeRequest(EffectSpec):
    power: float = Field(0.8, gt=0, lt=1)


class PowerRequest(EffectSpec):
    n_per_group: int = Field(..., gt=1)


class PowerCurveRequest(EffectSpec):
    power: float = Field(0.8, gt=0, lt=1)


# ---------- multiple testing ----------
class MultipleTestingRequest(BaseModel):
    p_values: list[Prob] = Field(..., min_length=1)
    method: Literal["bonferroni", "benjamini_hochberg"] = "benjamini_hochberg"
    alpha: float = Field(0.05, gt=0, lt=1)
    labels: Optional[list[str]] = None


# ---------- sequential ----------
class BoundariesRequest(BaseModel):
    n_looks: int = Field(5, ge=2, le=20)
    looks: Optional[list[float]] = Field(None, description="Information fractions, e.g. [0.2,0.4,0.6,0.8,1]")
    alpha: float = Field(0.05, gt=0, lt=1)
    spending: Spending = "obrien_fleming"


class SequentialAnalyzeRequest(GroupSpec):
    planned_total_n: int = Field(..., gt=10, description="Total rows (both arms) at the final look")
    n_looks: int = Field(5, ge=2, le=20)
    looks: Optional[list[float]] = None
    spending: Spending = "obrien_fleming"


class PeekingRequest(BaseModel):
    n_looks: int = Field(5, ge=2, le=20)
    alpha: float = Field(0.05, gt=0, lt=1)
    n_sims: int = Field(20_000, ge=1000, le=200_000)
    seed: Optional[int] = 42


# ---------- causal (observational) ----------
class PSMRequest(BaseModel):
    dataset_id: str
    treat_col: str
    outcome_col: str
    covariates: list[str] = Field(..., min_length=1)
    caliper_sd: float = Field(0.05, ge=0, le=5, description="Max logit-PS distance, in SDs (0 = no caliper). Smaller = stricter")
    n_neighbors: int = Field(1, ge=1, le=10)
    n_boot: int = Field(200, ge=20, le=1000)
    squares: bool = False
    zero_flags: bool = False
    alpha: float = Field(0.05, gt=0, lt=1)
    seed: int = 0


class PSMSensitivityRequest(BaseModel):
    dataset_id: str
    treat_col: str
    outcome_col: str
    covariates: list[str] = Field(..., min_length=1)
    caliper_sd: float = Field(0.05, ge=0, le=5)
    n_neighbors: int = Field(1, ge=1, le=10)


class DIDRequest(BaseModel):
    dataset_id: str
    treat_col: str
    pre_col: str
    post_col: str
    covariates: Optional[list[str]] = None
    placebo_pre_col: Optional[str] = Field(None, description="Two EARLIER periods to test for pre-existing trends")
    placebo_post_col: Optional[str] = None
    alpha: float = Field(0.05, gt=0, lt=1)


class DoWhyRequest(BaseModel):
    dataset_id: str
    treat_col: str
    outcome_col: str
    covariates: list[str] = Field(..., min_length=1)
    refute_with: Literal["linear_regression", "propensity_score_weighting", "propensity_score_matching"] = "linear_regression"
    n_simulations: int = Field(25, ge=5, le=200, description="Simulations per refutation test (more = slower, steadier)")
    alpha: float = Field(0.05, gt=0, lt=1)
    seed: int = 0


class EconMLRequest(BaseModel):
    dataset_id: str
    treat_col: str
    outcome_col: str
    covariates: list[str] = Field(..., min_length=1)
    effect_modifiers: Optional[list[str]] = Field(None, description="Subset of covariates the effect may vary by (default: all)")
    n_trees: int = Field(500, ge=100, le=2000)
    min_propensity: float = Field(0.05, gt=0, lt=0.5, description="Clip propensity scores to [p, 1-p] in the doubly robust estimator")
    alpha: float = Field(0.05, gt=0, lt=1)
    seed: int = 0


# ---------- Bayesian A/B ----------
class BayesABRequest(GroupSpec):
    metric: Literal["auto", "binary", "continuous"] = "auto"
    prior_alpha: float = Field(1.0, gt=0, description="Beta prior (binary only); 1,1 = flat")
    prior_beta: float = Field(1.0, gt=0)
    draws: int = Field(2000, ge=200, le=10_000)
    tune: int = Field(1000, ge=200, le=5_000)
    chains: int = Field(4, ge=2, le=4)
    min_effect: float = Field(0.0, description="Also report P(difference > this value)")
    likelihood: Literal["normal", "student_t"] = Field("normal", description="Continuous outcomes only")
    seed: int = 0
