"""One way to render a page, so no route forgets the context every template needs.

Jinja2 autoescaping is on — Starlette's `Jinja2Templates` enables it for `.html` — which
is what stops a store name or an email address rendering as markup. Nothing in this app
uses `| safe`, and nothing should: the moment one template does, every value reaching it
becomes a potential injection and the audit surface stops being one line.

`page` exists so that the signed-in user is in the context of every template
automatically. Passing it by hand from each route works until one route forgets, and then
the navigation bar silently claims nobody is logged in.
"""

from __future__ import annotations

from typing import Any

from fastapi import Request, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from . import auth, config

templates = Jinja2Templates(directory=str(config.TEMPLATES_DIR))


def page(
    request: Request,
    template: str,
    context: dict[str, Any] | None = None,
    *,
    status_code: int = status.HTTP_200_OK,
) -> HTMLResponse:
    """Render `template` with the signed-in user always available as `user`."""
    merged: dict[str, Any] = {"user": auth.signed_in_user(request)}
    merged.update(context or {})
    return templates.TemplateResponse(request, template, merged, status_code=status_code)
