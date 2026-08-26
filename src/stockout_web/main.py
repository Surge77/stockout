"""The app: middleware, startup, routers, and one error page instead of a traceback.

`stockout_web` is a separate package from `stockout` on purpose. The library is pure and a
web app is nothing but I/O; the app imports the library and the library has never heard of
the app.

    uvicorn stockout_web.main:app --reload
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.sessions import SessionMiddleware

from stockout.errors import StockoutError

from . import config, db, users
from .routers import admin, forecast, products
from .routers import auth as auth_routes
from .service import ModelService
from .templating import page


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    """Create the accounts table and load the models before the first request.

    Fitting takes seconds and the artefact is large, so a per-request load would make
    every page pay for it. Failing here rather than lazily means a broken artefact stops
    the process at startup instead of turning into a 500 on somebody's first forecast.
    """
    db.initialise()
    _seed_admin()
    app.state.models = ModelService.load()

    # Warm the comparison tables in the background. Fitting every registered model for
    # both tasks takes about a minute and a half, almost all of it `svc`. Done here the
    # app still answers immediately and the admin page is usually ready before anybody
    # opens it; done lazily the first visitor waits the whole time; done synchronously
    # above, every restart does.
    warm = asyncio.create_task(_warm_comparisons(app))
    try:
        yield
    finally:
        warm.cancel()


async def _warm_comparisons(app: FastAPI) -> None:
    """Precompute the comparison tables, and never take the app down if it fails."""
    try:
        await app.state.models.comparison_tables()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        # Broad on purpose: a warm-up is an optimisation, and no failure in it is worth
        # taking the app down for. The page still computes on demand.
        print(f"comparison warm-up skipped: {exc}")


def _seed_admin() -> None:
    """Create the administrator named by the environment, if it is not there already.

    `config.seed_admin` and the registration docstring both described this account as the
    supported way to get an admin on a fresh install, and nothing called it — leaving
    "whoever registers first" as the only route to the admin module, which is not
    something an operator can arrange in advance.

    Idempotent: an existing address is left exactly as it is, including its role. Silently
    promoting an account because the environment changed would be a privilege escalation
    driven by a config file.
    """
    credentials = config.seed_admin()
    if credentials is None:
        return

    email, password = credentials
    with db.session() as connection:
        if users.by_email(connection, email) is not None:
            return
        try:
            users.create(connection, email=email, password=password, role="admin")
        except users.UserError as exc:
            # A refused seed must not take the process down: the app is still usable, and
            # the reason is something the operator needs to read rather than a traceback.
            print(f"seed admin not created: {exc}")


app = FastAPI(title="stockout — retail sales and product demand prediction", lifespan=lifespan)

# Signs the session cookie. httpOnly and eight hours, both from config.
app.add_middleware(
    SessionMiddleware,
    secret_key=config.session_secret(),
    max_age=config.SESSION_MAX_AGE_SECONDS,
    same_site="lax",
    # `secure_cookies()` existed and was never read, so `STOCKOUT_WEB_HTTPS=1` did
    # nothing and the session cookie went out without `Secure` even behind TLS.
    https_only=config.secure_cookies(),
)

app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")

app.include_router(auth_routes.router)
app.include_router(forecast.router)
app.include_router(products.router)
app.include_router(admin.router)


@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException) -> HTMLResponse:
    """Unauthenticated requests go to the sign-in page; everything else gets one line."""
    if exc.status_code == status.HTTP_401_UNAUTHORIZED:
        return RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)  # type: ignore[return-value]
    return page(
        request,
        "error.html",
        # `error.html` reads `code` and `message`. Jinja renders an undefined name as the
        # empty string, so the old `status`/`detail` keys produced a page with an empty
        # heading and an empty red box on every 403 and 404.
        {"code": exc.status_code, "message": exc.detail},
        status_code=exc.status_code,
    )


@app.exception_handler(StockoutError)
async def domain_error(request: Request, exc: StockoutError) -> HTMLResponse:
    """A domain error is a message for the reader, never a stack trace."""
    return page(
        request,
        "error.html",
        {"code": status.HTTP_400_BAD_REQUEST, "message": str(exc)},
        status_code=status.HTTP_400_BAD_REQUEST,
    )
