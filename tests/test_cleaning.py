import numpy as np
import pytest

from hotel_ml.clean import clean
from hotel_ml.validate import assert_valid, run_checks


def test_ingest_combines_hotels_and_snake_cases(raw_df):
    assert set(raw_df["hotel"]) == {"Resort", "City"}
    assert "is_canceled" in raw_df.columns and "adr" in raw_df.columns
    assert len(raw_df) == 9


def test_impossible_rows_dropped(clean_df):
    guests = clean_df["adults"] + clean_df["children"] + clean_df["babies"]
    assert (guests > 0).all()
    assert clean_df["adr"].between(0, 1000).all()
    assert len(clean_df) == 6  # 9 raw - zero guests - negative ADR - extreme ADR


def test_padding_and_null_strings_removed(clean_df):
    assert clean_df["meal"].isin({"BB", "HB", "FB", "SC"}).all()
    assert clean_df["agent"].isna().any()
    assert not (clean_df["agent"] == "NULL").any()
    assert clean_df["deposit_type"].isin({"No Deposit", "Non Refund", "Refundable"}).all()


def test_undefined_meal_mapped_and_country_imputed(clean_df):
    row = clean_df[clean_df["lead_time"] == 120].iloc[0]
    assert row["meal"] == "SC"
    assert row["country"] == "UNK"


def test_namibia_country_code_preserved(clean_df):
    assert "NA" in set(clean_df["country"])


def test_missing_children_imputed(clean_df):
    assert clean_df["children"].notna().all()
    assert clean_df["children"].dtype == np.int64


def test_duplicates_kept_and_flagged(clean_df):
    assert (clean_df["dup_group_size"] == 2).sum() == 2


def test_cleaning_log_counts_every_rule(raw_df, cfg):
    _, log = clean(raw_df, cfg)
    d = log.to_frame().set_index("rule")["rows"]
    assert d["zero_guests"] == 1
    assert d["negative_adr"] == 1
    assert d["extreme_adr"] == 1
    assert d["exact_duplicates"] == 1


def test_contract_passes_on_clean_and_fails_on_dirty(clean_df):
    assert_valid(clean_df)
    dirty = clean_df.copy()
    dirty.loc[0, "adr"] = -1
    dirty.loc[1, "meal"] = "BB   "
    report = run_checks(dirty)
    assert not report["passed"].all()
    with pytest.raises(ValueError):
        assert_valid(dirty)
