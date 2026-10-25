import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hotel_ml.config import load_config  # noqa: E402

RAW_HEADER = ("IsCanceled,LeadTime,ArrivalDateYear,ArrivalDateMonth,ArrivalDateWeekNumber,"
              "ArrivalDateDayOfMonth,StaysInWeekendNights,StaysInWeekNights,Adults,Children,Babies,"
              "Meal,Country,MarketSegment,DistributionChannel,IsRepeatedGuest,PreviousCancellations,"
              "PreviousBookingsNotCanceled,ReservedRoomType,AssignedRoomType,BookingChanges,DepositType,"
              "Agent,Company,DaysInWaitingList,CustomerType,ADR,RequiredCarParkingSpaces,"
              "TotalOfSpecialRequests,ReservationStatus,ReservationStatusDate")

# Hand-written rows that reproduce every kind of mess found in the real extract.
RESORT_ROWS = [
    # normal row, padded fields, NULL agent/company
    "0,342,2015,July,27,1,0,2,2,0,0,BB       ,PRT,Direct,Direct,              0,0,0,C               ,C               ,3,No Deposit     ,       NULL,       NULL,0,Transient,75.5,0,0,Check-Out,2015-07-03",
    # 'Undefined' meal, missing country, canceled
    "1,120,2015,August,33,10,1,2,2,0,0,Undefined,NULL,Online TA,TA/TO,              0,1,0,A               ,A               ,0,Non Refund     ,          9,       NULL,0,Transient,110,0,1,Canceled,2015-06-01",
    # zero guests -> must be dropped
    "0,5,2015,August,33,12,0,1,0,0,0,BB       ,GBR,Direct,Direct,              0,0,0,A               ,A               ,0,No Deposit     ,       NULL,       NULL,0,Transient,50,0,0,Check-Out,2015-08-13",
    # exact duplicate pair (group block)
    "1,60,2015,September,36,5,0,3,2,0,0,HB       ,ESP,Groups,TA/TO,              0,0,0,D               ,D               ,0,No Deposit     ,          1,         40,0,Transient-Party,90,0,0,Canceled,2015-08-01",
    "1,60,2015,September,36,5,0,3,2,0,0,HB       ,ESP,Groups,TA/TO,              0,0,0,D               ,D               ,0,No Deposit     ,          1,         40,0,Transient-Party,90,0,0,Canceled,2015-08-01",
]
CITY_ROWS = [
    # missing children ('NA')
    "0,10,2016,March,10,1,0,2,2,NA,0,BB,FRA,Online TA,TA/TO,0,0,0,A,A,0,No Deposit,9,NULL,0,Transient,95,0,1,Check-Out,2016-03-03",
    # negative ADR -> drop
    "0,3,2016,March,10,2,0,1,1,0,0,SC,PRT,Direct,Direct,0,0,0,A,A,0,No Deposit,NULL,NULL,0,Transient,-6.38,0,0,Check-Out,2016-03-03",
    # extreme ADR -> drop
    "1,30,2016,March,12,15,0,1,2,0,0,BB,PRT,Offline TA/TO,TA/TO,0,0,0,A,A,0,Non Refund,12,NULL,0,Transient,5400,0,0,Canceled,2016-02-20",
    # Namibia ('NA' country code) must survive cleaning
    "0,40,2017,May,19,8,1,1,2,0,0,BB,NA,Online TA,TA/TO,0,0,0,A,A,0,No Deposit,9,NULL,0,Transient,120,0,2,Check-Out,2017-05-10",
]


@pytest.fixture
def cfg(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "H1.csv").write_text(RAW_HEADER + "\n" + "\n".join(RESORT_ROWS) + "\n")
    (raw / "H2.csv").write_text(RAW_HEADER + "\n" + "\n".join(CITY_ROWS) + "\n")
    c = load_config()
    c["paths"] = {**c["paths"], "raw_dir": raw}
    return c


@pytest.fixture
def raw_df(cfg):
    from hotel_ml.ingest import load_raw
    return load_raw(cfg["paths"]["raw_dir"], cfg["raw_files"])


@pytest.fixture
def clean_df(raw_df, cfg):
    from hotel_ml.clean import clean
    df, _ = clean(raw_df, cfg)
    return df
