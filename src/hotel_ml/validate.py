"""Declarative data contract for the cleaned table.

A tiny, dependency-free stand-in for tools like pandera / Great Expectations:
each check returns the number of violating rows, and the pipeline refuses to
train on data that breaks the contract.
"""
from __future__ import annotations

import pandas as pd

ALLOWED = {
    "hotel": {"City", "Resort"},
    "meal": {"BB", "HB", "FB", "SC"},
    "deposit_type": {"No Deposit", "Non Refund", "Refundable"},
    "customer_type": {"Transient", "Transient-Party", "Contract", "Group"},
    "is_canceled": {0, 1},
    "is_repeated_guest": {0, 1},
}
NON_NEGATIVE = [
    "lead_time", "stays_in_weekend_nights", "stays_in_week_nights", "adults",
    "children", "babies", "previous_cancellations", "previous_bookings_not_canceled",
    "days_in_waiting_list", "total_of_special_requests", "adr",
]
NOT_NULL = ["hotel", "is_canceled", "lead_time", "arrival_date", "adr", "country",
            "meal", "deposit_type", "customer_type", "children"]


def run_checks(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for col in NOT_NULL:
        results.append(("not_null", col, int(df[col].isna().sum())))
    for col, allowed in ALLOWED.items():
        results.append(("allowed_values", col, int((~df[col].isin(allowed)).sum())))
    for col in NON_NEGATIVE:
        results.append(("non_negative", col, int((df[col] < 0).sum())))
    guests = df["adults"] + df["children"] + df["babies"]
    results.append(("at_least_one_guest", "adults+children+babies", int((guests == 0).sum())))
    pad = 0
    for c in df.select_dtypes(include=["object", "string"]).columns:
        s = df[c].dropna().astype(str)  # NaN != NaN, so drop before comparing
        pad += int((s != s.str.strip()).sum())
    results.append(("no_padding", "all text columns", pad))
    out = pd.DataFrame(results, columns=["check", "column", "violations"])
    out["passed"] = out["violations"] == 0
    return out


def assert_valid(df: pd.DataFrame) -> pd.DataFrame:
    report = run_checks(df)
    failed = report[~report["passed"]]
    if len(failed):
        raise ValueError(f"Data contract violated:\n{failed.to_string(index=False)}")
    return report
