"""One prediction, phrased for somebody deciding how much to order.

The library speaks in r², terciles and class labels. `Verdict` is the whole translation
into a sentence a shopkeeper can act on, plus the geometry of the quiet/typical/busy bar
that answers "is that a lot?" without the reader holding three numbers in their head.

Nothing here computes anything new. It renames, compares and phrases what the models
already produced — and where it makes a claim about how good a number is, that claim comes
from the held-out score rather than from an adjective.

Split out of `presentation.py`, which now keeps the per-entity profiles this reads.
"""

from __future__ import annotations

from dataclasses import dataclass

from stockout.targets import HIGH, LOW, MEDIUM

from .presentation import StoreProfile

_PERCENT = 100.0

#: Headroom added to each end of the scale bar, as a fraction of its span. Enough that the
#: outer bands are always visible; small enough that the marker still sits where the
#: numbers say it should.
_AXIS_PADDING = 0.15

#: What each demand class means for somebody deciding how much to order.
DEMAND_WORDS: dict[str, dict[str, str]] = {
    LOW: {
        "title": "Quiet day",
        "advice": "Below this store's usual. Ordering the normal amount risks leftover stock.",
        "tone": "quiet",
    },
    MEDIUM: {
        "title": "Typical day",
        "advice": "An ordinary day for this store. The usual order should be about right.",
        "tone": "typical",
    },
    HIGH: {
        "title": "Busy day",
        "advice": "Above this store's usual. Order extra so you do not run out.",
        "tone": "busy",
    },
}


@dataclass(frozen=True)
class Verdict:
    """One prediction, phrased."""

    sales: float
    demand_class: str
    profile: StoreProfile
    spread_low: float
    spread_high: float

    @property
    def title(self) -> str:
        return DEMAND_WORDS[self.demand_class]["title"]

    @property
    def advice(self) -> str:
        return DEMAND_WORDS[self.demand_class]["advice"]

    @property
    def tone(self) -> str:
        return DEMAND_WORDS[self.demand_class]["tone"]

    @property
    def implied_class(self) -> str:
        """The class the predicted *number* falls into, by this store's own cut points.

        Not the same thing as `demand_class`, which is what the classifier said. Two
        models, two answers, and nothing forces them to agree.
        """
        if self.sales < self.profile.quiet_cut:
            return LOW
        if self.sales > self.profile.busy_cut:
            return HIGH
        return MEDIUM

    @property
    def models_disagree(self) -> bool:
        return self.implied_class != self.demand_class

    @property
    def disagreement_text(self) -> str:
        """Said plainly, because the panel would otherwise contradict itself.

        `advice` is written from the classifier's label and `comparison_text` from the
        regressor's number. When the two models split, those two lines say opposite
        things one after the other — "below this store's usual" directly above "about the
        same as this store's usual" — which reads as a broken page rather than as the
        honest disagreement it is. Naming it turns a bug report into the point.
        """
        if not self.models_disagree:
            return ""
        return (
            f"The two models disagree here. {self.sales:,.0f} falls in "
            f"{self.implied_class} by this store's cut points, while the demand "
            f"classifier calls it {self.demand_class}. Neither corrects the other — "
            f"treat the day as borderline."
        )

    @property
    def difference_percent(self) -> float:
        if not self.profile.typical_sales:
            return 0.0
        return (self.sales - self.profile.typical_sales) / self.profile.typical_sales * _PERCENT

    @property
    def comparison_text(self) -> str:
        """How this compares with the store's own usual day, in one sentence."""
        difference = self.difference_percent
        usual = f"{self.profile.typical_sales:,.0f}"
        if abs(difference) < 5:
            return f"About the same as this store's usual {usual} a day."
        direction = "more than" if difference > 0 else "less than"
        return f"{abs(difference):,.0f}% {direction} this store's usual {usual} a day."

    @property
    def arrow(self) -> str:
        difference = self.difference_percent
        if abs(difference) < 5:
            return "="
        return "up" if difference > 0 else "down"

    def _axis(self) -> tuple[float, float]:
        """The range the bar spans, with room beyond both cut points.

        Taking the axis straight from `min`/`max` of the values collapses an outer band
        to zero width whenever nothing exceeds its cut point — which is the normal case.
        A prediction below `quiet_cut` puts `busy_cut` at exactly 100%, the busy band
        renders zero pixels wide, and the bar silently claims this store is never busy.
        The padding keeps all three bands on screen so the marker is read against a
        complete scale rather than a truncated one.
        """
        values = (
            self.sales,
            self.spread_low,
            self.spread_high,
            self.profile.quiet_cut,
            self.profile.busy_cut,
        )
        low, high = min(values), max(values)
        span = high - low
        padding = span * _AXIS_PADDING if span > 0 else max(abs(high) * _AXIS_PADDING, 1.0)
        return low - padding, high + padding

    def _position(self, value: float) -> float:
        """`value` as a 0-100 percentage of the axis."""
        low, high = self._axis()
        span = high - low
        if span <= 0:
            return 50.0
        return max(0.0, min(_PERCENT, (value - low) / span * _PERCENT))

    def scale_position(self) -> float:
        """Where the prediction sits on this store's own range, as a 0-100 percentage.

        The bar shows quiet / typical / busy as three bands and marks the prediction on
        them, which answers "is that a lot?" without the reader having to hold three
        numbers in their head.
        """
        return self._position(self.sales)

    def band_edges(self) -> tuple[float, float]:
        """The quiet/typical and typical/busy boundaries, as percentages of the bar."""
        return self._position(self.profile.quiet_cut), self._position(self.profile.busy_cut)

    @property
    def spread_text(self) -> str:
        width = self.spread_high - self.spread_low
        return (
            f"The models range from {self.spread_low:,.0f} to {self.spread_high:,.0f} "
            f"— a spread of {width:,.0f}."
        )
