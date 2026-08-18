"""The application: middleware, routers, and what happens once at startup.

Run it with:

    pip install -e ".[web]"
    uvicorn stockout_web.main:app --reload

Startup does three things and each is a decision rather than boilerplate. It creates the
users table if absent, seeds the first admin from the environment when one is configured,
and loads the model artefact **once** — because a fitted pipeline is tens of megabytes and
loading it per request would be invisible in any single response and ruinous across all of
them.

The app starts even with no model and no accounts. That is the first-run state: an admin
has to be able to sign in before there is anything to serve, so a missing artefact becomes
a message on the page rather than a crash at boot.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

from . import config, db, users
from .routers import admin, forecast
from .routers import auth as auth_routes
from .service import ModelService
from .templating import page


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    db.initialise()
    _seed_admin()

    app.state.models = ModelService()
    app.state.models.load()
    yield


def _seed_admin() -> None:
    """Create the configured admin if the table is empty. Never overwrites an account.

    Only on an empty table, deliberately. Seeding into a populated one would let anybody
    who can set an environment variable reset the admin password on a running system, and
    would silently resurrect an account somebody had deactivated on purpose.
    """
    credentials = config.seed_admin()
    if credentials is None:
        return

    email, password = credentials
    with db.session() as connection:
        if users.count(connection) > 0:
            return
        try:
            users.create(connection, email=email, password=password, role="admin")
        except users.UserError:
            # A bad seed password must not stop the app booting — registration still
            # works, and the first account created that way becomes the admin.
            return


def create_app() -> FastAPI:
    app = FastAPI(
        title="stockout",
        description="Demand forecasting and demand classification for retail stores.",
        version="0.6.0",
        lifespan=lifespan,
    )

    # Signs the session cookie. `https_only` follows STOCKOUT_WEB_HTTPS because a Secure
    # cookie is never sent over plain http, and the default here is http://127.0.0.1.
    app.add_middleware(
        SessionMiddleware,
        secret_key=config.session_secret(),
        max_age=config.SESSION_MAX_AGE_SECONDS,
        same_site="lax",
        https_only=config.secure_cookies(),
    )
    # Never `["*"]`, and never with credentials allowed against a wildcard. This app runs
    # on a laptop and has no business accepting a cross-site request from anywhere else.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.ALLOWED_ORIGINS),
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")
    app.include_router(auth_routes.router)
    app.include_router(forecast.router)
    app.include_router(admin.router)

    _install_error_pages(app)

    @app.get("/health", include_in_schema=False)
    async def health(request: Request) -> dict[str, object]:
        """Liveness plus whether a model is actually loaded. No account required.

        Deliberately says nothing about *which* model or where it came from: a health
        check is read by anything that can reach the port.
        """
        return {"status": "ok", "model_loaded": request.app.state.models.state.is_ready}

    return app


def _install_error_pages(app: FastAPI) -> None:
    """Turn the two framework exceptions into pages a person can read.

    A 401 becomes a redirect to the login form rather than a JSON body, because the
    caller is a browser following a link. Everything else keeps its status code and gains
    a page — and none of them ever renders the exception's own text for a 500, which is
    where a path or a query would leak.
    """

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> Response:
        if exc.status_code == status.HTTP_401_UNAUTHORIZED:
            return RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)
        return page(
            request,
            "error.html",
            {"code": exc.status_code, "message": str(exc.detail)},
            status_code=exc.status_code,
        )

    @app.exception_handler(RequestValidationError)
    async def invalid_form(request: Request, exc: RequestValidationError) -> HTMLResponse:
        """A malformed form is the user's mistake, so name the field rather than 500.

        Only the field names are shown, never the submitted values: echoing input back
        into a page is how a reflected payload gets there.
        """
        fields = sorted({str(error["loc"][-1]) for error in exc.errors()})
        return page(
            request,
            "error.html",
            {"code": 422, "message": f"Check these fields: {', '.join(fields)}."},
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        )


app = create_app()
