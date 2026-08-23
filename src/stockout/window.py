"""Which dates the deployed models may be asked about, and what an answer means there.

Every feature in `features.py` is built from the calendar or from store metadata. Nothing
is derived by shifting the sales column, so scoring a date needs no history of that date —
the models are mechanically able to answer any date at all. Which dates to *accept* is
therefore a policy question rather than a technical one, and this module is where the
policy lives: once, for the CLI and the web panel both, which previously disagreed. The
panel refused out-of-window dates and the CLI did not check at all.

Three regions, and the distinction between the last two is the point of the module.

**Before the training window.** Refused. A date earlier than any row the fit ever saw
describes a period with no data behind it in either direction.

**Inside the training window.** Answered, and marked `observed`. The true figure for that
store-day is in the dataset. A prediction here is not a forecast; it is the model being
checked against something already known — which is what makes it worth showing beside the
actual, and what makes it worthless as a product on its own.

**After the training window.** Answered, and marked `forecast`. The only region where a
prediction is not redundant with a lookup, and the only region where no measured score
stands behind it. Both facts travel on the status so a caller can print them.

`observed` is the honest default for a demonstration and `forecast` is the honest default
for a use. Refusing the second, as this package did until now, left only the region where
`SELECT sales WHERE store = ? AND date = ?` beats every model in the registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from .errors import StockoutError

Region = Literal["observed", "forecast"]


@dataclass(frozen=True)
class DateStatus:
    """A parsed request date, and what the models can honestly claim about it."""

    stamp: pd.Timestamp
    region: Region
    observed_from: pd.Timestamp
    observed_through: pd.Timestamp

    @property
    def is_forecast(self) -> bool:
        return self.region == "forecast"

    @property
    def days_beyond(self) -> int:
        """How far past the end of the data this is. Zero inside the window."""
        if not self.is_forecast:
            return 0
        return int((self.stamp - self.observed_through).days)

    def caveat(self) -> str | None:
        """The sentence a surface should show beside a forecast, or None inside."""
        if not self.is_forecast:
            return None
        return (
            f"{self.days_beyond:,} days past the end of the data "
            f"({self.observed_through:%Y-%m-%d}). No measured score covers this period, "
            f"and the store-age features are held at their end-of-data values."
        )


def parse(date: str | pd.Timestamp) -> pd.Timestamp:
    """A date, or a `StockoutError` naming what could not be read."""
    try:
        stamp = pd.to_datetime(date)
    except (ValueError, TypeError) as exc:
        raise StockoutError(f"{date!r} is not a date") from exc
    if pd.isna(stamp):
        raise StockoutError(f"{date!r} is not a date")
    return pd.Timestamp(stamp)


def check(
    date: str | pd.Timestamp,
    observed_from: pd.Timestamp,
    observed_through: pd.Timestamp,
) -> DateStatus:
    """Classify a request date against the span of the data behind the models.

    Forward is allowed and backward is not, and the asymmetry is deliberate. A date after
    the window is the question the package exists to answer, and every feature needed to
    answer it is knowable in advance. A date before the window is not a harder version of
    the same question — there is simply nothing there.
    """
    stamp = parse(date)
    if stamp < observed_from:
        raise StockoutError(
            f"{stamp:%Y-%m-%d} is before the data, which starts "
            f"{observed_from:%Y-%m-%d}. There is nothing behind that period."
        )
    region: Region = "forecast" if stamp > observed_through else "observed"
    return DateStatus(
        stamp=stamp,
        region=region,
        observed_from=observed_from,
        observed_through=observed_through,
    )
