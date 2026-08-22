"""The app: middleware, startup, routers, and one error page instead of a traceback.

`stockout_web` is a separate package from `stockout` on purpose. The library is pure and a
web app is nothing but I/O; the app imports the library and the library has never heard of
the app.

    uvicorn stockout_web.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.sessions import SessionMiddleware

from stockout.errors import StockoutError

from . import config, db
from .routers import admin, forecast
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
    app.state.models = ModelService.load()
    yield


app = FastAPI(title="stockout — retail sales and product demand prediction", lifespan=lifespan)

# Signs the session cookie. httpOnly and eight hours, both from config.
app.add_middleware(
    SessionMiddleware,
    secret_key=config.session_secret(),
    max_age=config.SESSION_MAX_AGE_SECONDS,
    same_site="lax",
    https_only=False,
)

app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")

app.include_router(auth_routes.router)
app.include_router(forecast.router)
app.include_router(admin.router)


@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException) -> HTMLResponse:
    """Unauthenticated requests go to the sign-in page; everything else gets one line."""
    if exc.status_code == status.HTTP_401_UNAUTHORIZED:
        return RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)  # type: ignore[return-value]
    return page(
        request,
        "error.html",
        {"status": exc.status_code, "detail": exc.detail},
        status_code=exc.status_code,
    )


@app.exception_handler(StockoutError)
async def domain_error(request: Request, exc: StockoutError) -> HTMLResponse:
    """A domain error is a message for the reader, never a stack trace."""
    return page(
        request,
        "error.html",
        {"status": status.HTTP_400_BAD_REQUEST, "detail": str(exc)},
        status_code=status.HTTP_400_BAD_REQUEST,
    )
