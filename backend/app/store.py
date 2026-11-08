"""In-memory dataset store. Fine for a demo; data is lost when the server restarts."""
import json
import uuid

import pandas as pd
from fastapi import HTTPException

from app.schemas import ColumnInfo, DatasetInfo

_STORE: dict[str, pd.DataFrame] = {}


def register(df: pd.DataFrame, name: str, notes=None, true_effect=None) -> DatasetInfo:
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")].copy()  # drop stray index columns
    dataset_id = uuid.uuid4().hex[:12]
    _STORE[dataset_id] = df
    cols = [ColumnInfo(name=str(c), dtype=str(df[c].dtype), n_unique=int(df[c].nunique()),
                       n_missing=int(df[c].isna().sum())) for c in df.columns]
    preview = json.loads(df.head(5).to_json(orient="records"))
    return DatasetInfo(dataset_id=dataset_id, name=name, n_rows=len(df), columns=cols,
                       preview=preview, notes=notes, true_effect=true_effect)


def get(dataset_id: str) -> pd.DataFrame:
    if dataset_id not in _STORE:
        raise HTTPException(404, f"Dataset '{dataset_id}' not found (the server may have restarted). Upload it again.")
    return _STORE[dataset_id]
