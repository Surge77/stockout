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
from ..described import describe
from ..service import ModelService
from ..templating import page

router = APIRouter()

MAX_COMPARISON_MODELS = 12

#: The kinds of day the source distinguishes, offered in the order they read.
#:
#: These meanings are the competition's own, published beside the data, so naming them
#: is not the invention that section 6 of TODO.md forbids for the shop grouping.
#:
#: This was a checkbox, which collapsed all three holidays onto `a`. Christmas trades
#: nothing like an ordinary bank holiday and the fitted models already carry the
#: distinction — the form was the only place it was being thrown away.
STATE_HOLIDAYS: tuple[tuple[str, str], ...] = (
    ("0", "An ordinary trading day"),
    ("a", "Public holiday"),
    ("b", "Easter holiday"),
    ("c", "Christmas"),
)

_HOLIDAY_CODES = frozenset(code for code, _ in STATE_HOLIDAYS)


def holiday_code(raw: str) -> str:
    """The submitted kind of day, or a refusal.

    A `<select>` constrains a browser, not a request. Without this an unrecognised code
    would travel to `features.LABEL_VOCABULARIES`, be folded into `UNKNOWN`, and come
    back as a confident prediction for a day nobody described.
    """
    code = raw.strip().lower()
    if code not in _HOLIDAY_CODES:
        offered = ", ".join(repr(known) for known, _ in STATE_HOLIDAYS)
        raise StockoutError(f"{raw!r} is not a kind of day; expected one of {offered}")
    return code


def _service(request: Request) -> ModelService:
    """The one instance, built at startup and kept on the app."""
    return request.app.state.models


def _context(request: Request) -> dict[str, object]:
    """What the form needs to render: the stores and the window it may ask about."""
    service = _service(request)
    start, end = service.date_bounds()
    return {
        # Profiles, not bare ids. "1" tells a reader nothing about what they are picking;
        # "Store 1 — Type A, Basic range, usually about 9,300 a day" tells them what kind
        # of shop it is and what an ordinary day there looks like, which is the context
        # every number further down the page is relative to.
        "options": service.store_options(),
        "state_holidays": STATE_HOLIDAYS,
        "entity_label": service.entity_label(),
        # A product deployment picks a shop and then an item. One dropdown of every
        # store-item pair would be five hundred entries where two are ten and fifty.
        "has_products": service.has_products(),
        "unit": service.unit(),
        "uses_promotions": service.catalogue.uses_promotions(),
        "stores": service.stores() if service.has_products() else [],
        "date_min": start,
        # The last day with a true figure. No longer the picker's `max` — the input is
        # open-ended forward now, and this is the line the page draws between "checkable
        # against an actual" and "a forecast nothing has scored".
        "observed_through": end,
        "dataset": service.artifact.dataset,
        "regressor": service.artifact.regressor_name,
        "classifier": service.artifact.classifier_name,
    }


@router.get("/", response_class=HTMLResponse)
async def home(request: Request, user: CurrentUser) -> HTMLResponse:
    """The forecast form. Requires an account; `current_user` turns absence into a 401."""
    context = _context(request)
    service = _service(request)
    if service.has_products():
        # The item list belongs to a shop, and no shop is chosen yet — so open on the
        # first one, which is also what the shop dropdown will be showing.
        first = service.stores()[0]
        context.update({"store": first, "items": service.item_options(first)})
    return page(request, "forecast.html", context)


@router.post("/forecast", response_class=HTMLResponse)
async def make_forecast(
    request: Request,
    user: CurrentUser,
    store: Annotated[str, Form()],
    date: Annotated[str, Form()],
    item: Annotated[str | None, Form()] = None,
    promo: Annotated[bool, Form()] = False,
    school_holiday: Annotated[bool, Form()] = False,
    state_holiday: Annotated[str, Form()] = "0",
) -> HTMLResponse:
    """Answer one entity-day with both models, or explain why it cannot be answered."""
    context = _context(request)
    # Every submitted field goes back into the context, including the checkboxes. A form
    # that silently drops one on re-render tells the user their last answer came from
    # inputs they can no longer see.
    service = _service(request)
    context.update(
        {
            "store": store,
            "item": item,
            "date": date,
            "promo": promo,
            "school_holiday": school_holiday,
            "state_holiday": state_holiday,
            "items": service.item_options(store) if service.has_products() else [],
        }
    )
    entity = service.entity_for(store, item)

    try:
        answer = service.forecast(
            entity=entity,
            date=date,
            promo=promo,
            school_holiday=school_holiday,
            state_holiday=holiday_code(state_holiday),
        )
    except StockoutError as exc:
        context["error"] = str(exc)
        return page(request, "forecast.html", context)

    low, high = answer.thresholds
    # The true figure, where the dataset has one. Showing it beside the prediction is the
    # answer to "why not just look it up" — inside the window that is exactly what this
    # page now does, and putting the error on screen is more honest than hiding the
    # comparison the visitor could make anyway.
    actual = service.actual_for(entity, answer.date)
    context.update(
        {
            "actual": None if actual is None else f"{actual:,.0f}",
            "error_amount": None if actual is None else f"{answer.sales - actual:+,.0f}",
            "error_pct": (
                None if not actual else f"{(answer.sales - actual) / actual:+.1%}"
            ),
            "status": answer.status,
            "caveat": answer.status.caveat() if answer.status else None,
            "answer": answer,
            # The same prediction, phrased for somebody deciding how much to order. The
            # models speak in terciles and r²; `Verdict` is the whole translation.
            "verdict": service.verdict(answer),
            "sales": f"{answer.sales:,.0f}",
            "demand_class": answer.demand_class,
            "low_cut": f"{low:,.0f}",
            "high_cut": f"{high:,.0f}",
            "per_model": sorted(
                answer.per_model.items(), key=lambda pair: pair[1], reverse=True
            )[:MAX_COMPARISON_MODELS],
            "deployed": service.artifact.regressor_name,
        }
    )
    return page(request, "forecast.html", context)


#: What the described-store form offers, in words rather than in the source's codes.
#:
#: The assortment meanings are documented by the competition, so those labels are its own.
#: The store grouping is **not**: Rossmann sorted its shops into four kinds and never said
#: what the kinds are. Naming them here would read better and be invented, so the letter
#: stays and the form explains what it is — the shop's owner considered the grouping
#: meaningful, and shops within a group trade alike, which is the whole reason a model
#: that has never seen your shop can say anything about it at all.
STORE_TYPES: tuple[tuple[str, str], ...] = (
    ("a", "Group A — the commonest kind"),
    ("b", "Group B — rare, and much busier than the rest"),
    ("c", "Group C"),
    ("d", "Group D"),
)
ASSORTMENTS: tuple[tuple[str, str], ...] = (
    ("a", "Basic — the smallest range of goods"),
    ("b", "Extra — a wider range"),
    ("c", "Extended — the widest range"),
)


def _whatif_context(request: Request) -> dict[str, object]:
    service = _service(request)
    start, end = service.date_bounds()
    return {
        # False where the source publishes nothing describing an entity, so no cold-start
        # model exists to answer with. The page explains that rather than offering a form
        # whose every answer would be the same number.
        "available": service.supports_description(),
        "dataset": service.artifact.dataset,
        "entity_label": service.entity_label(),
        "store_types": STORE_TYPES,
        "assortments": ASSORTMENTS,
        "state_holidays": STATE_HOLIDAYS,
        "date_min": start,
        "observed_through": end,
        "unseen_r2": service.artifact.cold_scores.get("unseen_store_r2"),
        "cold_features": service.artifact.cold_feature_columns,
        # Which form to render. A described item is its category and nothing else; a
        # described shop has four attributes. Both come from what the source publishes.
        "has_products": service.has_products(),
        "unit": service.unit(),
        "categories": service.categories(),
    }


@router.get("/what-if", response_class=HTMLResponse)
async def whatif_form(request: Request, user: CurrentUser) -> HTMLResponse:
    """Describe a shop that does not exist and ask what it would take."""
    return page(request, "whatif.html", _whatif_context(request))


@router.post("/what-if", response_class=HTMLResponse)
async def whatif(
    request: Request,
    user: CurrentUser,
    store_type: Annotated[str, Form()] = "",
    assortment: Annotated[str, Form()] = "",
    category: Annotated[str, Form()] = "",
    competition_distance: Annotated[str, Form()] = "",
    promo2: Annotated[bool, Form()] = False,
    date: Annotated[str, Form()] = "",
    promo: Annotated[bool, Form()] = False,
    state_holiday: Annotated[str, Form()] = "0",
    school_holiday: Annotated[bool, Form()] = False,
) -> HTMLResponse:
    """Answer for a described shop, with what each attribute was worth."""
    context = _whatif_context(request)
    context.update({
        "store_type": store_type, "assortment": assortment, "category": category,
        "competition_distance": competition_distance, "promo2": promo2,
        "date": date, "promo": promo,
        "state_holiday": state_holiday, "school_holiday": school_holiday,
    })

    try:
        answer, why = describe(
            _service(request).artifact,
            store_type=store_type,
            assortment=assortment,
            category=category,
            competition_distance=_metres(competition_distance),
            promo2=promo2,
            date=date,
            promo=promo,
            state_holiday=holiday_code(state_holiday),
            school_holiday=school_holiday,
        )
    except StockoutError as exc:
        context["error"] = str(exc)
        return page(request, "whatif.html", context)

    context.update({
        "answer": answer,
        "sales": f"{answer.sales:,.0f}",
        "why": [(label, f"{value:+,.0f}") for label, value in why],
        "caveat": answer.status.caveat() if answer.status else None,
    })
    return page(request, "whatif.html", context)


def _metres(raw: str) -> float | None:
    """An empty distance box means "no competitor recorded", not zero."""
    text = raw.strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError as exc:
        raise StockoutError(f"{raw!r} is not a distance in metres") from exc
    if value < 0:
        raise StockoutError("a distance cannot be negative")
    return value
