"""The admin module: what is deployed, retrain it, compare everything, manage accounts.

Every route here depends on `AdminUser`, which re-reads the role from the database on each
request rather than trusting the session cookie. A demoted or deactivated admin therefore
loses access on their very next request rather than at their next sign-in.
"""

from __future__ import annotations

import asyncio
import sqlite3
from typing import Annotated, Literal

import pandas as pd
from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse

from stockout.data.adapters import SOURCES, available
from stockout.errors import StockoutError
from stockout.evaluate import to_markdown
from stockout.registry import model_names

from .. import db, users
from ..auth import AdminUser
from ..service import ModelService
from ..templating import page
from ..warmup import warm_service

router = APIRouter(prefix="/admin")


def _service(request: Request) -> ModelService:
    return request.app.state.models


def _dashboard_context(request: Request) -> dict[str, object]:
    service = _service(request)
    artifact = service.artifact
    start, end = service.date_bounds()
    return {
        "cards": service.cards(),
        "artifact": artifact,
        "provenance": artifact.provenance().splitlines(),
        "regressors": model_names("regression"),
        "classifiers": model_names("classification"),
        # Only sources whose files are actually on disk. Offering `rossmann` on a machine
        # that has never run `fetch` would put a choice in the form whose only outcome is
        # an error, and the dashboard is where somebody goes to change what is deployed.
        "datasets": _deployable(),
        "dataset": artifact.dataset,
        "entity_label": artifact.entity_label,
        "entity_count": len(artifact.entities),
        "date_min": start,
        "date_max": end,
    }


@router.get("", response_class=HTMLResponse)
async def dashboard(request: Request, admin: AdminUser) -> HTMLResponse:
    return page(request, "admin.html", _dashboard_context(request))


def _deployable() -> list[dict[str, object]]:
    """The sources that can be deployed from here, with what choosing one costs.

    The cost is worth stating on the form. Fitting `synthetic_products` is half a minute;
    fitting `rossmann` is over an hour, because the cold-start score alone refits the
    regressor once per held-out shop over 670,422 rows. A browser will not wait for that,
    so the form says which is which and points the long one at the CLI.
    """
    out: list[dict[str, object]] = []
    for source in SOURCES:
        if not available(source):
            continue
        out.append(
            {
                "name": source.name,
                "description": source.description,
                "kind": "product" if source.has_products else "store-day",
                "slow": bool(source.raw_files),
            }
        )
    return out


def _deployable_names() -> set[str]:
    return {str(source["name"]) for source in _deployable()}


@router.post("/train", response_class=HTMLResponse)
async def retrain(
    request: Request,
    admin: AdminUser,
    regressor: Annotated[str, Form()],
    classifier: Annotated[str, Form()],
    dataset: Annotated[str | None, Form()] = None,
) -> HTMLResponse:
    """Fit a new pair and deploy it, or explain why the request was refused.

    `dataset` is what makes the product pages reachable without the CLI. Deploying a
    product source swaps `/what-if` for `/products` on the very next request, because the
    surfaces read the artefact rather than a setting — see `templating._provenance`.

    Defaults to what is already deployed, so a form that omits the field keeps its old
    meaning of "refit this data with a different pair".
    """
    service = _service(request)
    chosen = dataset or service.artifact.dataset

    # A `<select>` constrains a browser, not a request. Without this an unknown name would
    # reach `prepare` and surface as a stack trace rather than a refusal.
    if chosen not in _deployable_names():
        context = _dashboard_context(request)
        offered = ", ".join(sorted(_deployable_names()))
        context["error"] = f"{chosen!r} is not a deployable dataset; choose from {offered}"
        return page(request, "admin.html", context)

    try:
        replacement = await service.retrain(
            regressor=regressor, classifier=classifier, dataset=chosen
        )
    except StockoutError as exc:
        context = _dashboard_context(request)
        context["error"] = str(exc)
        return page(request, "admin.html", context)

    # Swap the whole object: a request arriving mid-retrain sees the old pair or the
    # new one, never a regressor from one and thresholds from the other.
    request.app.state.models = replacement

    # The comparison regressors are not fitted by `retrain` — see `ModelService.warm`.
    # Warmed here in the background so the dashboard comes back now rather than after
    # another seven fits on the new data.
    request.app.state.warm = asyncio.create_task(warm_service(replacement))

    context = _dashboard_context(request)
    context["notice"] = f"deployed {regressor} and {classifier} on {chosen}"
    return page(request, "admin.html", context)


#: Columns `evaluate.compare` reports that this page does not show.
#:
#: `taught` says whether an estimator appears in the course notebooks. That governs what
#: may be added to this repository — `scripts/check_taught.py` still enforces it, and the
#: CLI and README still report it — but the page asks which model predicts sales best, and
#: where an estimator was learnt has no bearing on that.
NOT_SHOWN: tuple[str, ...] = ("taught",)


def _for_display(table: pd.DataFrame) -> pd.DataFrame:
    """The frame as the page shows it.

    Applied before both renderings. The markdown block underneath the tables prints every
    column of whatever frame it is handed, so dropping the column in the template alone
    would have taken it out of one place and left it in the other.
    """
    return table.drop(columns=[column for column in NOT_SHOWN if column in table.columns])


@router.get("/comparison", response_class=HTMLResponse)
async def comparison(request: Request, admin: AdminUser) -> HTMLResponse:
    """Every registered model on one split, for both tasks."""
    service = _service(request)
    regression, classification, floor = await service.comparison_tables()
    regression, classification = _for_display(regression), _for_display(classification)
    return page(
        request,
        "comparison.html",
        {
            "regression_rows": regression.to_dict("records"),
            "classification_rows": classification.to_dict("records"),
            "regression_markdown": to_markdown(regression),
            "classification_markdown": to_markdown(classification),
            "floor": f"{floor:.1%}",
            "dataset": service.artifact.dataset,
        },
    )


@router.get("/users", response_class=HTMLResponse)
async def user_list(request: Request, admin: AdminUser) -> HTMLResponse:
    with db.session() as connection:
        return page(request, "users.html", _accounts_context(connection))


@router.post("/users", response_class=HTMLResponse)
async def create_user(
    request: Request,
    admin: AdminUser,
    email: Annotated[str, Form()],
    password: Annotated[str, Form()],
    role: Annotated[Literal["user", "admin"], Form()] = "user",
) -> HTMLResponse:
    return await _mutate(
        request, lambda c: users.create(c, email=email, password=password, role=role)
    )


@router.post("/users/{user_id}/role", response_class=HTMLResponse)
async def change_role(
    request: Request,
    admin: AdminUser,
    user_id: int,
    role: Annotated[Literal["user", "admin"], Form()],
) -> HTMLResponse:
    return await _mutate(request, lambda c: users.set_role(c, user_id=user_id, role=role))


@router.post("/users/{user_id}/active", response_class=HTMLResponse)
async def change_active(
    request: Request,
    admin: AdminUser,
    user_id: int,
    is_active: Annotated[str, Form()],
) -> HTMLResponse:
    """The form posts `is_active` as the word, matching `users.set_active`'s own argument.

    It used to require a field called `active` holding `"1"`, which the template never
    sent — so the deactivate button answered 422 while looking perfectly fine.
    """
    wanted = is_active.strip().lower() in {"1", "true", "yes", "on"}
    return await _mutate(
        request, lambda c: users.set_active(c, user_id=user_id, is_active=wanted)
    )


async def _mutate(request: Request, operation) -> HTMLResponse:  # type: ignore[no-untyped-def]
    """Apply one account change and re-render the list, with any refusal shown."""
    error = None
    with db.session() as connection:
        try:
            operation(connection)
        except users.UserError as exc:
            error = str(exc)
        context = _accounts_context(connection)
    context["error"] = error
    return page(request, "users.html", context)


def _accounts_context(connection: sqlite3.Connection) -> dict[str, object]:
    """Everything `users.html` reads.

    Built in one place because it was previously assembled twice, by hand, with the wrong
    key both times: the routes passed `rows` while the template looped over `accounts`, and
    `roles` was never passed at all — so the table rendered headers over nothing and both
    role dropdowns came out with no options. Jinja renders an undefined name as the empty
    string, so neither failed loudly.
    """
    return {"accounts": users.listing(connection), "roles": list(users.ROLES)}
