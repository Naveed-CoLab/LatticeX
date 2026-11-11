"""Walk-forward folds with an embargo, plus the locked final holdout.

Fold for test year Y:
  train: every date up to `embargo_days` trading days before 1 Jan Y (expanding window)
  test:  every trading date in year Y, before the holdout starts
Rows are also dropped if their target is only known on/after the holdout start.
"""
from dataclasses import dataclass

import pandas as pd


@dataclass
class Fold:
    year: int
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_folds(dates, first_test_year: int, last_test_year: int, embargo_days: int,
               holdout_start: str) -> list[Fold]:
    dates = pd.DatetimeIndex(sorted(pd.unique(pd.Series(dates))))
    holdout = pd.Timestamp(holdout_start)
    folds = []
    for y in range(first_test_year, last_test_year + 1):
        test = dates[(dates >= f"{y}-01-01") & (dates < f"{y + 1}-01-01") & (dates < holdout)]
        if len(test) == 0:
            continue
        i = dates.get_loc(test[0])
        if i - embargo_days - 1 < 0:
            continue
        folds.append(Fold(y, dates[i - embargo_days - 1], test[0], test[-1]))
    return folds


def eval_dates(dates, step: int) -> pd.DatetimeIndex:
    """Every `step`-th trading day: the shared scoring grid for every model."""
    d = pd.DatetimeIndex(sorted(pd.unique(pd.Series(dates))))
    return d[::step]
