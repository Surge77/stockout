"""The admin module: retrain, read the comparison, manage accounts.

Every route here depends on `AdminUser`, which is `current_user` plus a role check. The
dependency is the authorisation — no handler below re-checks a role, because a check
written per handler is a check that will eventually be missing from one of them, and that
one will be the interesting route.

Two of these are slow on purpose. Training fits both tasks on every row; the comparison
fits twelve regressors or nine classifiers. Both are pushed off the event loop by
`ModelService` so that a user asking for a forecast is not queued behind an admin's
curiosity.

**There is no model upload and there is no data upload.** An admin picks from the registry
by name, and the artefact is written by this process from data already on disk. Accepting
a `.joblib` over HTTP would be arbitrary code execution — `joblib.load` runs what it reads
— and it is not a feature worth having at that price.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Form, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse

from stockout.errors import StockoutError
from stockout.models.registry import model_names
from stockout.train import DEFAULT_CLASSIFIER, DEFAULT_REGRESSOR

from .. import db, users
from ..auth import AdminUser
from ..service import ModelService
from ..templating import page

router = APIRouter(prefix="/admin", tags=["admin"])


def _service(request: Request) -> ModelService:
    return request.app.state.models


def _dashboard_context(request: Request) -> dict[str, object]:
    state = _service(request).state
    return {
        "artifact": state.artifact,
        "ready": state.is_ready,
        "load_error": state.error,
        "stores": state.stores,
        "last_date": state.last_date,
        "regressors": model_names("regression"),
        "classifiers": model_names("classification"),
        "default_regressor": DEFAULT_REGRESSOR,
        "default_classifier": DEFAULT_CLASSIFIER,
    }


@router.get("", response_class=HTMLResponse)
async def dashboard(request: Request, admin: AdminUser) -> HTMLResponse:
    """Provenance of the deployed artefact, and the controls for replacing it."""
    return page(request, "admin.html", _dashboard_context(request))


@router.post("/train", response_class=HTMLResponse)
async def retrain(
    request: Request,
    admin: AdminUser,
    regressor: Annotated[str, Form()] = DEFAULT_REGRESSOR,
    classifier: Annotated[str, Form()] = DEFAULT_CLASSIFIER,
) -> HTMLResponse:
    """Fit both tasks on everything, save, and reload what every user is served.

    Awaited rather than fired into the background, because an admin who is told "training
    started" and given no way to learn whether it finished has been told nothing. It runs
    in a worker thread, so the process keeps serving while it does.
    """
    context = _dashboard_context(request)
    try:
        artifact = await _service(request).retrain(regressor=regressor, classifier=classifier)
    except StockoutError as exc:
        context["error"] = str(exc)
        return page(request, "admin.html", context, status_code=status.HTTP_400_BAD_REQUEST)

    context = _dashboard_context(request)  # reload: the artefact has just changed
    context["trained"] = artifact.summary()
    return page(request, "admin.html", context)


@router.get("/comparison", response_class=HTMLResponse)
async def comparison(
    request: Request,
    admin: AdminUser,
    task: str = "regression",
    models: Annotated[list[str] | None, Query()] = None,
) -> HTMLResponse:
    """Every registered model scored on one held-out window, simplest first.

    Registry order rather than sorted by score: a table sorted by score answers *which
    won*, and this order also answers *did the extra complexity pay*.

    `models` narrows it to a subset. The whole registry is twelve fits for regression and
    nine for classification, which is a slow page to reload while you are looking at one
    comparison — and an unknown name is refused by `ModelService` rather than skipped.
    """
    if task not in ("regression", "classification"):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "task must be regression or classification"
        )

    try:
        table = await _service(request).comparison(task, models)  # type: ignore[arg-type]
    except StockoutError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    return page(
        request,
        "comparison.html",
        {
            "task": task,
            "columns": [str(column) for column in table.columns],
            "rows": table.to_dict(orient="records"),
            "metric": "wmape" if task == "regression" else "macro_f1",
        },
    )


@router.get("/users", response_class=HTMLResponse)
async def user_list(request: Request, admin: AdminUser) -> HTMLResponse:
    with db.session() as connection:
        accounts = users.listing(connection)
    return page(request, "users.html", {"accounts": accounts, "roles": users.ROLES})


@router.post("/users", response_class=HTMLResponse)
async def create_user(
    request: Request,
    admin: AdminUser,
    email: Annotated[str, Form()],
    password: Annotated[str, Form()],
    role: Annotated[str, Form()] = "user",
) -> HTMLResponse:
    """Add an account, at either role. The only way an admin is made after the first."""
    return await _mutate(request, lambda c: users.create(
        c, email=email, password=password, role=role,  # type: ignore[arg-type]
    ))


@router.post("/users/{user_id}/role", response_class=HTMLResponse)
async def change_role(
    request: Request, admin: AdminUser, user_id: int, role: Annotated[str, Form()] = "user"
) -> HTMLResponse:
    return await _mutate(request, lambda c: users.set_role(
        c, user_id=user_id, role=role,  # type: ignore[arg-type]
    ))


@router.post("/users/{user_id}/active", response_class=HTMLResponse)
async def change_active(
    request: Request, admin: AdminUser, user_id: int, is_active: Annotated[bool, Form()] = False
) -> HTMLResponse:
    """Deactivate or restore. Deletion is deliberately not offered — see `users.py`."""
    return await _mutate(request, lambda c: users.set_active(
        c, user_id=user_id, is_active=is_active,
    ))


async def _mutate(request: Request, operation) -> HTMLResponse:  # type: ignore[no-untyped-def]
    """Apply one change to the user table, then re-render the list.

    A rejected change — a duplicate email, the last admin demoting themselves — comes
    back as the same page with the reason on it rather than as a redirect that loses the
    message.
    """
    try:
        with db.session() as connection:
            operation(connection)
    except users.UserError as exc:
        with db.session() as connection:
            accounts = users.listing(connection)
        return page(
            request,
            "users.html",
            {"accounts": accounts, "roles": users.ROLES, "error": str(exc)},
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    return RedirectResponse("/admin/users", status_code=status.HTTP_303_SEE_OTHER)  # type: ignore[return-value]
