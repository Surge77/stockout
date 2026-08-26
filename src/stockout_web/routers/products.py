"""The shelf: every item one shop stocks on one day, ranked by what it will shift.

The forecast page answers one series at a time, which is the right unit for a model and the
wrong unit for a person. Nobody orders one item. This page answers the question somebody
actually has — *what should I order more of on Tuesday* — by scoring the whole shelf in one
pass and sorting it.

Only available where the deployed artefact has a product axis. On a store-day deployment
there is no shelf to rank, and the page says that rather than 404ing: a visitor who
followed a link deserves to know the deployment changed, not that the URL is wrong.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse

from stockout.errors import StockoutError

from ..auth import CurrentUser
from ..service import ModelService
from ..templating import page
from .forecast import STATE_HOLIDAYS, holiday_code

router = APIRouter()

#: How many rows the table shows before the rest are folded away. A shelf is long, and the
#: top of the list is the part that changes an order.
DEFAULT_SHOWN = 25


def _service(request: Request) -> ModelService:
    return request.app.state.models


def _context(request: Request) -> dict[str, object]:
    """What the form needs, plus whether this deployment can answer at all."""
    service = _service(request)
    start, end = service.date_bounds()
    return {
        "available": service.has_products(),
        "dataset": service.artifact.dataset,
        "entity_label": service.entity_label(),
        "unit": service.unit(),
        "stores": service.stores() if service.has_products() else [],
        "state_holidays": STATE_HOLIDAYS,
        "uses_promotions": service.catalogue.uses_promotions(),
        "date_min": start,
        "observed_through": end,
        "regressor": service.artifact.regressor_name,
        "classifier": service.artifact.classifier_name,
        # Shown beside the note about the two models disagreeing. Where the classifier is
        # barely above its floor, saying so is what tells a reader which number to trust —
        # and the floor is measured rather than assumed. Per-entity terciles put it near a
        # third by construction *in training*; on a chronological hold-out the residual
        # distribution drifts and it reached 45% here, which a hardcoded 33% would have
        # misreported as a much better classifier than it is.
        "classifier_accuracy": service.artifact.scores.get("classification", {}).get(
            "accuracy", 0.0
        ),
        "class_floor": service.benchmarks.class_floor(),
    }


@router.get("/products", response_class=HTMLResponse)
async def shelf_form(request: Request, user: CurrentUser) -> HTMLResponse:
    """Pick a shop and a day."""
    return page(request, "products.html", _context(request))


@router.post("/products", response_class=HTMLResponse)
async def shelf(
    request: Request,
    user: CurrentUser,
    store: Annotated[str, Form()],
    date: Annotated[str, Form()],
    promo: Annotated[bool, Form()] = False,
    school_holiday: Annotated[bool, Form()] = False,
    state_holiday: Annotated[str, Form()] = "0",
) -> HTMLResponse:
    """Rank the shelf, or explain why this deployment cannot."""
    context = _context(request)
    context.update(
        {
            "store": store,
            "date": date,
            "promo": promo,
            "school_holiday": school_holiday,
            "state_holiday": state_holiday,
        }
    )

    try:
        ranked = _service(request).rank_products(
            store=store,
            date=date,
            promo=promo,
            school_holiday=school_holiday,
            state_holiday=holiday_code(state_holiday),
        )
    except StockoutError as exc:
        context["error"] = str(exc)
        return page(request, "products.html", context)

    context.update(
        {
            "ranked": ranked,
            "rows": ranked.busiest(DEFAULT_SHOWN),
            "hidden": max(0, len(ranked) - DEFAULT_SHOWN),
            "total": f"{ranked.total:,.0f}",
            "caveat": ranked.status.caveat() if ranked.status else None,
            # The busiest item's prediction, so every bar can be drawn as a share of it.
            # Taken from the answer rather than recomputed in the template, where a
            # division by zero has nowhere to go.
            "peak": max((item.sales for item in ranked.items), default=0.0) or 1.0,
        }
    )
    return page(request, "products.html", context)
