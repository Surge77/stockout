"""The user module: ask for a store-day, get both answers.

One store-day in; predicted sales *and* the demand class out, with that store's own cut
points beside them. The two come from different models and can disagree, and a page
showing one while implying both would be the most misleading thing here.

The expandable section lists what every registered regressor predicted for the same row,
because they do not agree either and a single number implies a certainty nobody has.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse

from stockout.errors import StockoutError

from ..auth import CurrentUser
from ..service import ModelService
from ..templating import page

router = APIRouter()

MAX_COMPARISON_MODELS = 12


def _service(request: Request) -> ModelService:
    """The one instance, built at startup and kept on the app."""
    return request.app.state.models


def _context(request: Request) -> dict[str, object]:
    """What the form needs to render: the stores and the window it may ask about."""
    service = _service(request)
    start, end = service.date_bounds()
    return {
        "entities": service.entities(),
        "entity_label": service.entity_label(),
        "date_min": start,
        "date_max": end,
        "dataset": service.artifact.dataset,
        "regressor": service.artifact.regressor_name,
        "classifier": service.artifact.classifier_name,
    }


@router.get("/", response_class=HTMLResponse)
async def home(request: Request, user: CurrentUser) -> HTMLResponse:
    """The forecast form. Requires an account; `current_user` turns absence into a 401."""
    return page(request, "forecast.html", _context(request))


@router.post("/forecast", response_class=HTMLResponse)
async def make_forecast(
    request: Request,
    user: CurrentUser,
    store: Annotated[str, Form()],
    date: Annotated[str, Form()],
    promo: Annotated[bool, Form()] = False,
    school_holiday: Annotated[bool, Form()] = False,
) -> HTMLResponse:
    """Answer one store-day with both models, or explain why it cannot be answered."""
    context = _context(request)
    context.update({"store": store, "date": date, "promo": promo})

    try:
        answer = _service(request).forecast(
            entity=store, date=date, promo=promo, school_holiday=school_holiday
        )
    except StockoutError as exc:
        context["error"] = str(exc)
        return page(request, "forecast.html", context)

    low, high = answer.thresholds
    context.update(
        {
            "answer": answer,
            "sales": f"{answer.sales:,.0f}",
            "demand_class": answer.demand_class,
            "low_cut": f"{low:,.0f}",
            "high_cut": f"{high:,.0f}",
            "per_model": sorted(
                answer.per_model.items(), key=lambda pair: pair[1], reverse=True
            )[:MAX_COMPARISON_MODELS],
            "deployed": _service(request).artifact.regressor_name,
        }
    )
    return page(request, "forecast.html", context)
