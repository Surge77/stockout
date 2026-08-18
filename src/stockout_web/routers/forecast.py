"""The user module: ask for a store-day, get both answers.

This is the whole point of the app, and the reason both halves appear on one page.
`stockout` solves two problems on the same rows — **how much will this store sell**
(regression) and **is that a Low, Medium or High day for this shop** (classification) —
and a page that showed only the number would be serving half the model while implying it
was the whole thing.

The two answers can disagree, and when they do that is information rather than a bug: the
classifier optimises the boundary it is scored on, while the regressor minimises squared
error and is pulled towards busy days. `docs/results.md` measures the gap. The page shows
the class the classifier chose *and* the cut points the number is being judged against, so
a reader can see which side of the boundary they landed on and by how much.

Every failure here is a `StockoutError` with a message written for a person — "cannot
forecast 2016-06-01: store 1's history ends 2014-12-31..." — so it is shown rather than
swallowed into a generic 500.
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated

from fastapi import APIRouter, Form, Request, status
from fastapi.responses import HTMLResponse

from stockout.errors import StockoutError

from ..auth import CurrentUser
from ..service import ModelService
from ..templating import page

router = APIRouter(tags=["forecast"])


def _service(request: Request) -> ModelService:
    """The one instance, built at startup and kept on the app."""
    return request.app.state.models


def _context(request: Request) -> dict[str, object]:
    """What the form needs to render: the stores and the window it may ask about."""
    state = _service(request).state
    return {
        "ready": state.is_ready,
        "load_error": state.error,
        "stores": state.stores,
        "last_date": state.last_date,
        "servable_until": state.servable_until,
        "horizon": state.artifact.horizon if state.artifact else None,
        "artifact": state.artifact,
    }


@router.get("/", response_class=HTMLResponse)
async def home(request: Request, user: CurrentUser) -> HTMLResponse:
    """The forecast form. Requires an account; `current_user` turns absence into a 401."""
    return page(request, "forecast.html", _context(request))


@router.post("/forecast", response_class=HTMLResponse)
async def make_forecast(
    request: Request,
    user: CurrentUser,
    store: Annotated[int, Form()],
    when: Annotated[dt.date, Form(alias="date")],
    promo: Annotated[bool, Form()] = False,
    school_holiday: Annotated[bool, Form()] = False,
    is_open: Annotated[bool, Form()] = False,
) -> HTMLResponse:
    """Answer one store-day with both models, or explain why it cannot be answered."""
    context = _context(request)
    context.update({"store": store, "date": when, "promo": promo,
                    "school_holiday": school_holiday, "is_open": is_open})

    try:
        answer = _service(request).predict(
            store=store,
            when=when,
            promo=promo,
            school_holiday=school_holiday,
            is_open=is_open,
        )
    except StockoutError as exc:
        # These messages are written for a person and name the limit they hit, so they
        # are shown rather than replaced with something generic. Nothing in them reveals
        # a path, a stack frame or another user's data.
        context["error"] = str(exc)
        return page(request, "forecast.html", context, status_code=status.HTTP_400_BAD_REQUEST)

    low, high = answer.thresholds
    context["answer"] = {
        "sales": answer.sales,
        "demand_class": answer.demand_class,
        "low": low,
        "high": high,
        "store": answer.store,
        "date": answer.date.date(),
    }
    return page(request, "forecast.html", context)
