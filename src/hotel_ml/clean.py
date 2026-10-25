"""Turn the raw text extract into a typed, validated table.

Every rule records how many rows it touched, so the data-quality report is
generated from the code that actually ran instead of being written by hand.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

INT_COLS = [
    "is_canceled", "lead_time", "arrival_date_year", "arrival_date_week_number",
    "arrival_date_day_of_month", "stays_in_weekend_nights", "stays_in_week_nights",
    "adults", "babies", "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "booking_changes", "days_in_waiting_list",
    "required_car_parking_spaces", "total_of_special_requests",
]


@dataclass
class CleaningLog:
    entries: list[dict] = field(default_factory=list)

    def add(self, rule: str, action: str, rows: int, reason: str) -> None:
        self.entries.append({"rule": rule, "action": action, "rows": int(rows), "reason": reason})

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.entries)


def clean(raw: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, CleaningLog]:
    c = cfg["cleaning"]
    log = CleaningLog()
    df = raw.copy()

    # 1. Fixed-width padding from the source system
    obj = df.columns
    padded = sum(int((df[col] != df[col].str.strip()).sum()) for col in obj)
    df = df.apply(lambda s: s.str.strip())
    log.add("whitespace_padding", "strip", padded,
            "Source export pads fields to fixed width ('BB       ', '       NULL')")

    # 2. Literal 'NULL' / empty strings -> real missing values
    agent_null = int((df["agent"] == "NULL").sum())
    company_null = int((df["company"] == "NULL").sum())
    # Don't treat "NA" as missing globally: it is Namibia's ISO country code.
    df = df.replace({"NULL": np.nan, "": np.nan})
    log.add("agent_null", "flag -> 'none' category", agent_null,
            "NULL agent means booked without a travel agent (informative, not unknown)")
    log.add("company_null", "collapse to has_company flag", company_null,
            f"{company_null / len(df):.0%} missing; 350+ ids too sparse to learn from")

    # 3. Types
    for col in INT_COLS:
        df[col] = pd.to_numeric(df[col], errors="raise").astype("int64")
    df["children"] = pd.to_numeric(df["children"], errors="coerce")
    df["adr"] = pd.to_numeric(df["adr"], errors="raise")

    # 4. Missing children -> 0 (4 rows; all other guest fields present)
    n = int(df["children"].isna().sum())
    df["children"] = df["children"].fillna(0).astype("int64")
    log.add("children_missing", "impute 0", n, "Only 4 rows; adults recorded, no evidence of children")

    # 5. 'Undefined' meal is documented as equivalent to SC (self-catering)
    n = int((df["meal"] == "Undefined").sum())
    df["meal"] = df["meal"].replace("Undefined", "SC")
    log.add("meal_undefined", "map -> SC", n, "Data dictionary: Undefined and SC are the same package")

    for col in ["market_segment", "distribution_channel"]:
        n = int((df[col] == "Undefined").sum())
        df[col] = df[col].replace("Undefined", np.nan)
        log.add(f"{col}_undefined", "set missing", n, "Placeholder value with no meaning")

    n = int(df["country"].isna().sum())
    df["country"] = df["country"].fillna("UNK")
    log.add("country_missing", "impute 'UNK'", n, "Country unknown at booking time")

    # 6. Dates: build from parts; the raw week number is NOT ISO (drop it)
    df["arrival_date"] = pd.to_datetime(
        df["arrival_date_year"].astype(str) + "-" + df["arrival_date_month"] + "-"
        + df["arrival_date_day_of_month"].astype(str), format="%Y-%B-%d")
    iso = df["arrival_date"].dt.isocalendar().week.astype("int64")
    n = int((iso != df["arrival_date_week_number"]).sum())
    df = df.drop(columns=["arrival_date_week_number"])
    log.add("week_number_inconsistent", "drop column, derive from date", n,
            "Source week numbering disagrees with the ISO calendar for ~54% of rows")
    df["reservation_status_date"] = pd.to_datetime(df["reservation_status_date"])

    # 7. Impossible records -> drop
    guests = df["adults"] + df["children"] + df["babies"]
    rules = {
        "zero_guests": (guests == 0, "Booking with no guests"),
        "negative_adr": (df["adr"] < 0, "Negative nightly price"),
        "extreme_adr": (df["adr"] > c["adr_max"], f"ADR above {c['adr_max']} (5,400/night entry error)"),
        "implausible_kids": ((df["children"] > c["max_children"]) | (df["babies"] > c["max_babies"]),
                             "9-10 children/babies in a single room"),
    }
    drop = pd.Series(False, index=df.index)
    for name, (mask, reason) in rules.items():
        log.add(name, "drop row", int(mask.sum()), reason)
        drop |= mask
    df = df.loc[~drop].reset_index(drop=True)

    # 8. Duplicates: keep, but flag. See README "Duplicates" for why.
    # NOTE: dup_group_size is for evaluation only. The duplicate key includes the
    # label, so using it as a feature would leak the outcome.
    key = pd.util.hash_pandas_object(df, index=False)
    df["dup_group_size"] = key.map(key.value_counts()).astype("int64")
    log.add("exact_duplicates", "keep + flag (dup_group_size)", int(df.duplicated().sum()),
            "No booking ID in source; identical rows are mostly group blocks. "
            "Temporal split keeps each group on one side, so no train/test contamination")

    return df, log
