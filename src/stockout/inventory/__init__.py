"""The decision layer: turn a forecast into a replenishment order, then price the outcome.

`policy` derives the service level from the cost pair and the base-stock level from the
forecast; `simulate` walks the stock forward and prices what happened. The frontier those
two produce is the argument the whole project is making — that a forecast is judged by
the decision it drives, not by a percentage error.
"""

from __future__ import annotations
