import numpy as np
import pandas as pd
import yaml

from hotel_ml import business
from hotel_ml.config import PROJECT_ROOT
from hotel_ml.evaluate import expected_calibration_error, score
from hotel_ml.features import FEATURES, build_features
from hotel_ml.models import make_hgb, make_logreg


def test_features_have_expected_columns(clean_df):
    X = build_features(clean_df)
    assert list(X.columns) == FEATURES
    assert (X["total_nights"] == X["stays_in_weekend_nights"] + X["stays_in_week_nights"]).all()
    assert X["arrival_month"].between(1, 12).all()


def test_no_leakage_columns_in_features():
    leaky = set(yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))["leakage_columns"])
    leaky |= {"is_canceled", "dup_group_size"}
    assert not leaky & set(FEATURES)


def test_features_are_row_wise(clean_df):
    """Computing features on a subset must equal the subset of features on the full set."""
    full = build_features(clean_df)
    part = build_features(clean_df.iloc[:3])
    pd.testing.assert_frame_equal(full.iloc[:3], part)


def _synthetic(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({c: rng.normal(size=n) for c in FEATURES})
    for c in ["hotel", "meal", "country", "market_segment", "distribution_channel",
              "reserved_room_type", "deposit_type", "customer_type", "agent"]:
        X[c] = rng.choice(["a", "b", "c"], size=n)
    y = (X["lead_time"] + (X["deposit_type"] == "a") + rng.normal(scale=0.5, size=n) > 0.5).astype(int)
    return X, y.values


def test_models_fit_and_handle_unseen_categories():
    X, y = _synthetic()
    X_new = X.head(5).copy()
    X_new["country"] = "never-seen"
    for model in (make_logreg(min_freq=1), make_hgb({"max_leaf_nodes": 15}, min_freq=1, max_iter=50)):
        model.fit(X, y)
        p = model.predict_proba(X_new)[:, 1]
        assert np.all((p >= 0) & (p <= 1))
        assert score(y, model.predict_proba(X)[:, 1])["roc_auc"] > 0.8


def test_score_and_ece():
    y = np.array([0, 0, 1, 1])
    m = score(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["roc_auc"] == 1.0 and m["f1"] == 1.0
    assert expected_calibration_error(y, np.array([0.0, 0.0, 1.0, 1.0])) == 0.0


def test_business_value_logic():
    y = np.array([1, 0, 1])
    p = np.array([0.9, 0.9, 0.1])
    rev = np.array([400.0, 300.0, 1000.0])
    # flag first two: TP recovers 0.5*400, FP costs 250
    assert business.net_value(y, p, rev, 0.5, resell_rate=0.5, walk_cost=250) == 200 - 250
    t, v = business.best_threshold(y, p, rev, 0.5, 250)
    assert v >= 0  # never worse than doing nothing at a high enough threshold
