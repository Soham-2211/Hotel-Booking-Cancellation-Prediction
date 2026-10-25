"""Feature engineering.

All features here are row-wise and use only information available when the
booking is made, so they can be computed before the split without leakage.
Anything that needs to *learn* from data (rare-category grouping, scaling)
lives inside the sklearn Pipeline and is fit on training rows only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORICAL = [
    "hotel", "meal", "country", "market_segment", "distribution_channel",
    "reserved_room_type", "deposit_type", "customer_type", "agent",
]
NUMERIC = [
    "lead_time", "stays_in_weekend_nights", "stays_in_week_nights", "total_nights",
    "adults", "children", "babies", "total_guests", "is_family", "is_day_use",
    "is_repeated_guest", "previous_cancellations", "previous_bookings_not_canceled",
    "prev_cancel_ratio", "days_in_waiting_list", "adr", "adr_per_person",
    "booking_value", "total_of_special_requests", "has_company",
    "arrival_month", "arrival_month_sin", "arrival_month_cos", "arrival_dow",
    "booking_dow", "is_weekend_arrival", "booking_month",
]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "is_canceled"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for col in ["hotel", "meal", "country", "market_segment", "distribution_channel",
                "reserved_room_type", "deposit_type", "customer_type"]:
        X[col] = df[col].astype("object").where(df[col].notna(), "missing")
    X["agent"] = df["agent"].fillna("none").astype(str)

    X["lead_time"] = df["lead_time"]
    X["stays_in_weekend_nights"] = df["stays_in_weekend_nights"]
    X["stays_in_week_nights"] = df["stays_in_week_nights"]
    X["total_nights"] = X["stays_in_weekend_nights"] + X["stays_in_week_nights"]
    X["adults"], X["children"], X["babies"] = df["adults"], df["children"], df["babies"]
    X["total_guests"] = X["adults"] + X["children"] + X["babies"]
    X["is_family"] = ((X["children"] + X["babies"]) > 0).astype(int)
    X["is_day_use"] = (X["total_nights"] == 0).astype(int)

    X["is_repeated_guest"] = df["is_repeated_guest"]
    X["previous_cancellations"] = df["previous_cancellations"]
    X["previous_bookings_not_canceled"] = df["previous_bookings_not_canceled"]
    prev_total = X["previous_cancellations"] + X["previous_bookings_not_canceled"]
    X["prev_cancel_ratio"] = np.where(prev_total > 0, X["previous_cancellations"] / prev_total.clip(lower=1), np.nan)

    X["days_in_waiting_list"] = df["days_in_waiting_list"]
    X["adr"] = df["adr"]
    X["adr_per_person"] = df["adr"] / (X["adults"] + X["children"]).clip(lower=1)
    X["booking_value"] = df["adr"] * X["total_nights"]
    X["total_of_special_requests"] = df["total_of_special_requests"]
    X["has_company"] = df["company"].notna().astype(int)

    arrival = df["arrival_date"]
    booking = arrival - pd.to_timedelta(df["lead_time"], unit="D")
    X["arrival_month"] = arrival.dt.month
    X["arrival_month_sin"] = np.sin(2 * np.pi * X["arrival_month"] / 12)
    X["arrival_month_cos"] = np.cos(2 * np.pi * X["arrival_month"] / 12)
    X["arrival_dow"] = arrival.dt.dayofweek
    X["is_weekend_arrival"] = (X["arrival_dow"] >= 4).astype(int)
    X["booking_dow"] = booking.dt.dayofweek
    X["booking_month"] = booking.dt.month
    return X[FEATURES]
