"""End-to-end pipeline: raw CSVs -> cleaned data -> models -> reports.

Usage:  python -m hotel_ml.pipeline [--config config.yaml] [--fast]
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from . import business, plots
from .clean import clean
from .config import load_config
from .evaluate import score
from .features import CATEGORICAL, FEATURES, NUMERIC, TARGET, build_features
from .ingest import load_raw
from .models import make_hgb, make_logreg, param_grid
from .validate import assert_valid


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def temporal_split(df: pd.DataFrame, cfg: dict):
    d = df["arrival_date"]
    tr = d <= cfg["split"]["train_end"]
    va = (d > cfg["split"]["train_end"]) & (d <= cfg["split"]["valid_end"])
    te = d > cfg["split"]["valid_end"]
    return tr.values, va.values, te.values


def fit_predict(model, X_tr, y_tr, X_ev):
    model.fit(X_tr, y_tr)
    return model.predict_proba(X_ev)[:, 1]


def run(config_path=None, fast: bool = False) -> dict:
    cfg = load_config(config_path)
    P, seed, biz = cfg["paths"], cfg["random_state"], cfg["business"]
    min_freq = cfg["cleaning"]["rare_category_min_count"]
    results: dict = {}

    # ---------- 1. Ingest, clean, validate ----------
    raw = load_raw(P["raw_dir"], cfg["raw_files"])
    log(f"raw rows: {len(raw):,}")
    df, clog = clean(raw, cfg)
    contract = assert_valid(df)
    log(f"clean rows: {len(df):,}  (contract: {int(contract['passed'].sum())}/{len(contract)} checks pass)")
    df.to_csv(P["processed_dir"] / "bookings_clean.csv.gz", index=False)
    clog.to_frame().to_csv(P["reports_dir"] / "cleaning_log.csv", index=False)
    results["data"] = {"raw_rows": len(raw), "clean_rows": len(df),
                       "dropped_rows": len(raw) - len(df),
                       "exact_duplicate_rows": int(df.drop(columns="dup_group_size").duplicated().sum()),
                       "cancel_rate": round(float(df[TARGET].mean()), 4)}

    X, y = build_features(df), df[TARGET].values
    revenue = X["booking_value"].values
    tr, va, te = temporal_split(df, cfg)
    results["split"] = {k: {"rows": int(m.sum()), "cancel_rate": round(float(y[m].mean()), 4),
                            "from": str(df.loc[m, "arrival_date"].min().date()),
                            "to": str(df.loc[m, "arrival_date"].max().date())}
                        for k, m in [("train", tr), ("valid", va), ("test", te)]}
    log(f"split: {results['split']}")

    # ---------- 2. Model selection on the validation season ----------
    search = []
    for C in cfg["model"]["logreg_C"]:
        p = fit_predict(make_logreg(C, min_freq, seed), X[tr], y[tr], X[va])
        search.append({"model": "logreg", "params": {"C": C}, **score(y[va], p)})
    grid = param_grid(cfg["model"]["hgb_search"])
    if fast:
        grid = grid[:2]
    for params in grid:
        p = fit_predict(make_hgb(params, min_freq, cfg["model"]["max_iter"], seed), X[tr], y[tr], X[va])
        search.append({"model": "hgb", "params": params, **score(y[va], p)})
        log(f"hgb {params} valid AUC={search[-1]['roc_auc']}  logloss={search[-1]['log_loss']}")
    search_df = pd.DataFrame(search).sort_values("log_loss")
    search_df.to_csv(P["reports_dir"] / "model_search.csv", index=False)
    best_lr = min((s for s in search if s["model"] == "logreg"), key=lambda s: s["log_loss"])
    best_hgb = min((s for s in search if s["model"] == "hgb"), key=lambda s: s["log_loss"])
    results["selected"] = {"logreg": best_lr["params"], "hgb": best_hgb["params"]}

    # Threshold chosen on validation predictions of the train-only model, then frozen.
    hgb_trainonly = make_hgb(best_hgb["params"], min_freq, cfg["model"]["max_iter"], seed)
    p_va = fit_predict(hgb_trainonly, X[tr], y[tr], X[va])
    thr, val_value = business.best_threshold(y[va], p_va, revenue[va], biz["resell_rate"], biz["walk_cost"])
    results["threshold"] = {"value": round(thr, 3), "valid_net_value_eur": round(val_value)}
    log(f"business threshold (valid) = {thr:.2f}")

    # ---------- 3. Final models: refit on train+valid, evaluate once on test ----------
    trva = tr | va
    final_lr = make_logreg(best_lr["params"]["C"], min_freq, seed).fit(X[trva], y[trva])
    final_hgb = make_hgb(best_hgb["params"], min_freq, cfg["model"]["max_iter"], seed).fit(X[trva], y[trva])
    p_lr, p_hgb = final_lr.predict_proba(X[te])[:, 1], final_hgb.predict_proba(X[te])[:, 1]
    y_te = y[te]
    results["test"] = {
        "majority_baseline": score(y_te, np.full(len(y_te), y[trva].mean())),
        "logreg": score(y_te, p_lr, 0.5),
        "hgb@0.5": score(y_te, p_hgb, 0.5),
        "hgb@business_threshold": score(y_te, p_hgb, thr),
    }
    uniq = ~df.loc[te].drop(columns="dup_group_size").duplicated().values
    results["test"]["hgb_deduplicated_test"] = score(y_te[uniq], p_hgb[uniq], thr)
    for h in ["City", "Resort"]:
        m = (df.loc[te, "hotel"] == h).values
        results["test"][f"hgb_{h.lower()}"] = score(y_te[m], p_hgb[m], thr)
    for k, v in results["test"].items():
        log(f"test {k}: AUC={v['roc_auc']} PR-AUC={v['pr_auc']} brier={v['brier']}")

    # Business value on test
    nr = (X.loc[te, "deposit_type"] == "Non Refund").values.astype(float)
    results["business_test"] = {
        "assumptions": biz,
        "no_overbooking_eur": 0,
        "rule_overbook_all_non_refund_eur": round(business.net_value(y_te, nr, revenue[te], 0.5, **biz)),
        "logreg_eur": round(business.net_value(y_te, p_lr, revenue[te], thr, **biz)),
        "hgb_eur": round(business.net_value(y_te, p_hgb, revenue[te], thr, **biz)),
        "oracle_eur": round(business.net_value(y_te, y_te.astype(float), revenue[te], 0.5, **biz)),
        "bookings_flagged": int((p_hgb >= thr).sum()),
        "test_bookings": int(te.sum()),
    }
    log(f"business: {results['business_test']}")

    # ---------- 4. Leakage & evaluation-design experiments ----------
    hp = (best_hgb["params"], min_freq, cfg["model"]["max_iter"], seed)
    exp = [("Temporal split (reported result)", results["test"]["hgb@0.5"]["roc_auc"], "honest")]
    idx = np.arange(len(df))
    i_tr, i_te = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y)
    p = fit_predict(make_hgb(*hp), X.iloc[i_tr], y[i_tr], X.iloc[i_te])
    exp.append(("Random 80/20 split", round(float(score(y[i_te], p)["roc_auc"]), 4), "optimistic"))
    groups = pd.util.hash_pandas_object(df.drop(columns="dup_group_size"), index=False).values
    g_tr, g_te = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(X, y, groups))
    p = fit_predict(make_hgb(*hp), X.iloc[g_tr], y[g_tr], X.iloc[g_te])
    exp.append(("Random split, duplicates grouped", round(float(score(y[g_te], p)["roc_auc"]), 4), "optimistic"))

    leaky_cat = ["assigned_room_type"]
    leaky_num = ["required_car_parking_spaces", "booking_changes"]
    XL = X.copy()
    for c in leaky_cat + leaky_num:
        XL[c] = df[c].values
    m = make_hgb(*hp, categorical=CATEGORICAL + leaky_cat, numeric=NUMERIC + leaky_num)
    p = fit_predict(m, XL[trva], y[trva], XL[te])
    exp.append(("Temporal + post-booking columns", round(float(score(y_te, p)["roc_auc"]), 4), "leaky"))
    XS = X.copy()
    XS["reservation_status"] = df["reservation_status"].values
    m = make_hgb(*hp, categorical=CATEGORICAL + ["reservation_status"], numeric=NUMERIC)
    p = fit_predict(m, XS[trva], y[trva], XS[te])
    exp.append(("Temporal + reservation_status", round(float(score(y_te, p)["roc_auc"]), 4), "leaky"))
    results["evaluation_experiments"] = {e[0]: e[1] for e in exp}
    log(f"experiments: {results['evaluation_experiments']}")

    # ---------- 5. Explainability ----------
    rng = np.random.default_rng(seed)
    sample = rng.choice(np.where(te)[0], size=min(8000, int(te.sum())), replace=False)
    pi = permutation_importance(final_hgb, X.iloc[sample], y[sample], scoring="roc_auc",
                                n_repeats=3 if fast else 5, random_state=seed, n_jobs=-1)
    imp = (pd.DataFrame({"feature": FEATURES, "importance_mean": pi.importances_mean,
                         "importance_std": pi.importances_std})
           .sort_values("importance_mean", ascending=False))
    imp.to_csv(P["reports_dir"] / "permutation_importance.csv", index=False)
    results["top_features"] = imp.head(10)["feature"].tolist()

    # ---------- 6. Figures & artefacts ----------
    F = P["figures_dir"]
    plots.cleaning_impact(clog.to_frame(), F / "01_cleaning_impact.png")
    plots.cancel_over_time(df, F / "02_cancel_rate_over_time.png")
    plots.cancel_by_lead_time(df, F / "03_cancel_by_lead_time.png")
    plots.split_comparison(exp, F / "04_evaluation_design.png")
    plots.roc_pr(y_te, {"Logistic regression": p_lr, "Gradient boosting": p_hgb}, F / "05_roc_pr.png")
    plots.calibration(y_te, p_hgb, F / "06_calibration.png")
    plots.importance(imp, F / "07_permutation_importance.png")
    g, v = business.value_curve(y_te, p_hgb, revenue[te], biz["resell_rate"], biz["walk_cost"])
    plots.value_curve(g, v, thr, F / "08_business_value.png")
    # Hindsight only (never used for decisions): shows the cost of drift between seasons.
    results["threshold"]["test_optimal_in_hindsight"] = round(float(g[int(np.argmax(v))]), 3)
    results["threshold"]["test_value_at_hindsight_optimum_eur"] = round(float(v.max()))

    joblib.dump({"model": final_hgb, "threshold": thr, "features": FEATURES},
                P["models_dir"] / "hgb_cancellation.joblib")
    (P["reports_dir"] / "metrics.json").write_text(json.dumps(results, indent=2, default=str))
    write_quality_report(raw, df, clog, contract, P["reports_dir"] / "data_quality_report.md")
    log("done")
    return results


def write_quality_report(raw, df, clog, contract, path):
    raw_checks = run_checks_raw(raw)
    lines = ["# Data quality report", "",
             "_Generated by `hotel_ml.pipeline` - do not edit by hand._", "",
             f"- Raw rows: **{len(raw):,}** (H1 resort + H2 city)",
             f"- Clean rows: **{len(df):,}** ({len(raw) - len(df)} dropped)",
             f"- Columns: {raw.shape[1]} raw -> {df.shape[1]} clean", "",
             "## Issues found in the raw extract", "", raw_checks.pipe(md_table), "",
             "## Cleaning rules applied", "", clog.to_frame().pipe(md_table), "",
             "## Data contract on the cleaned table", "", contract.pipe(md_table), ""]
    path.write_text("\n".join(lines))


def md_table(d: pd.DataFrame) -> str:
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def run_checks_raw(raw: pd.DataFrame) -> pd.DataFrame:
    s = raw.apply(lambda c: c.str.strip())
    rows = [
        ("Fixed-width padded text cells", int(sum((raw[c] != s[c]).sum() for c in raw.columns))),
        ("Literal 'NULL' string cells", int((s == "NULL").sum().sum())),
        ("'Undefined' placeholder values", int((s == "Undefined").sum().sum())),
        ("Missing children (NA)", int((s["children"].isin(["NA", ""])).sum())),
        ("Missing country", int((s["country"].isin(["NULL", "NA", ""])).sum())),
        ("Exact duplicate rows", int(s.duplicated().sum())),
        ("Negative ADR", int((pd.to_numeric(s["adr"]) < 0).sum())),
        ("ADR > 1000", int((pd.to_numeric(s["adr"]) > 1000).sum())),
        ("Bookings with zero guests", int(((pd.to_numeric(s["adults"]) + pd.to_numeric(s["children"], errors="coerce").fillna(0)
                                            + pd.to_numeric(s["babies"])) == 0).sum())),
    ]
    return pd.DataFrame(rows, columns=["issue", "count"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--fast", action="store_true", help="tiny search grid for smoke tests")
    a = ap.parse_args()
    run(a.config, a.fast)


if __name__ == "__main__":
    main()
