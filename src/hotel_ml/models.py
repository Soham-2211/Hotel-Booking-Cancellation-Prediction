"""Model definitions. Every learned transformation lives inside the Pipeline."""
from __future__ import annotations

import itertools

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .features import CATEGORICAL, NUMERIC


def make_logreg(C: float = 1.0, min_freq: int = 30, seed: int = 42) -> Pipeline:
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_freq,
                              sparse_output=True), CATEGORICAL),
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                          ("scale", StandardScaler())]), NUMERIC),
    ])
    return Pipeline([("pre", pre),
                     ("clf", LogisticRegression(C=C, max_iter=3000, random_state=seed))])


def make_hgb(params: dict | None = None, min_freq: int = 30, max_iter: int = 600,
             seed: int = 42, categorical=CATEGORICAL, numeric=NUMERIC) -> Pipeline:
    params = params or {}
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                               encoded_missing_value=np.nan, min_frequency=min_freq,
                               max_categories=250), list(categorical)),
        ("num", "passthrough", list(numeric)),
    ])
    is_cat = [True] * len(categorical) + [False] * len(numeric)
    clf = HistGradientBoostingClassifier(
        categorical_features=is_cat, max_iter=max_iter, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=30, random_state=seed, **params)
    return Pipeline([("pre", pre), ("clf", clf)])


def param_grid(search: dict) -> list[dict]:
    keys = list(search)
    return [dict(zip(keys, vals)) for vals in itertools.product(*search.values())]
