"""Time-series-safe cross-validation splitters.

Standard k-fold CV shuffles rows and is *invalid* for financial time series: it
leaks future information into the training set and destroys autocorrelation
structure. We instead provide forward-chaining splits where every test block is
strictly in the future relative to its training block, separated by an
**embargo** gap to prevent leakage from the forward-looking target (the embargo
should be ``>=`` the prediction horizon).

Because the data is a *panel* (many tickers per date) splits are computed over
the unique sorted dates and then mapped back to row positions, so an entire
cross-section moves between train and test together.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_platform.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class TimeSeriesSplitter:
    """Forward-chaining splitter for panel data.

    Parameters
    ----------
    scheme:
        ``"expanding"`` grows the training set from the start each fold;
        ``"walk_forward"`` uses a fixed-size rolling training window.
    n_splits:
        Number of (train, test) folds. Test blocks tile the most recent
        ``n_splits * test_size`` dates.
    test_size:
        Number of unique dates per test block.
    min_train_size:
        Minimum number of unique training dates required for a fold to be
        emitted (also the rolling window length for ``walk_forward``).
    embargo:
        Number of dates skipped between the end of train and start of test.
    """

    scheme: str = "walk_forward"
    n_splits: int = 5
    test_size: int = 252
    min_train_size: int = 504
    embargo: int = 5

    def __post_init__(self) -> None:
        if self.scheme not in {"expanding", "walk_forward"}:
            raise ValueError(f"unknown scheme '{self.scheme}'")
        if self.n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        if self.test_size < 1:
            raise ValueError("test_size must be >= 1")

    def split(self, dates: pd.Series) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_positions, test_positions)`` integer index arrays.

        ``dates`` is the per-row date series of the (already date-sorted) frame.
        """
        dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
        unique_dates = np.sort(dates.unique())
        n_dates = len(unique_dates)

        needed = self.n_splits * self.test_size + self.min_train_size + self.embargo
        if n_dates < self.test_size + self.min_train_size + self.embargo:
            raise ValueError(
                f"Not enough dates ({n_dates}) for even one fold "
                f"(need >= {self.test_size + self.min_train_size + self.embargo})."
            )
        if n_dates < needed:
            logger.warning(
                "Only %d dates available; reducing effective folds to fit " "(requested %d).",
                n_dates,
                self.n_splits,
            )

        # Position of the first test date across all folds.
        first_test_pos = n_dates - self.n_splits * self.test_size
        # If that leaves too little training data, shrink the number of folds.
        date_to_pos = {d: i for i, d in enumerate(unique_dates)}
        row_pos = dates.map(date_to_pos).to_numpy()

        emitted = 0
        for i in range(self.n_splits):
            test_start = first_test_pos + i * self.test_size
            test_end = test_start + self.test_size
            train_end = test_start - self.embargo
            if train_end < self.min_train_size:
                continue
            if self.scheme == "expanding":
                train_start = 0
            else:  # walk_forward — fixed rolling window
                train_start = max(0, train_end - self.min_train_size)

            train_mask = (row_pos >= train_start) & (row_pos < train_end)
            test_mask = (row_pos >= test_start) & (row_pos < test_end)
            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            emitted += 1
            yield train_idx, test_idx

        if emitted == 0:
            raise ValueError(
                "No valid folds were produced; reduce min_train_size/test_size/n_splits."
            )

    def get_n_splits(self, dates: pd.Series) -> int:
        return sum(1 for _ in self.split(dates))
