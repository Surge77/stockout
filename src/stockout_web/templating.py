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
    merged: dict[str, Any] = {
        "user": auth.signed_in_user(request),
        "notice": auth.take_notice(request),
        **_provenance(request),
    }
    merged.update(context or {})
    return templates.TemplateResponse(request, template, merged, status_code=status_code)


def _provenance(request: Request) -> dict[str, Any]:
    """What the footer needs to describe the data honestly, on every page.

    Read from the loaded artefact rather than written into the templates. Both footers
    said "synthetic" long after the models had been retrained on real Rossmann data — a
    claim that is invisible to the test suite, survives every refactor, and is exactly the
    kind of stale sentence a reader is entitled to trust. Deriving it means retraining on
    a different source updates the wording by itself.

    Falls back to nothing when the app is running with a stubbed service, so the templates
    have to cope with absence anyway.
    """
    models = getattr(request.app.state, "models", None)
    artifact = getattr(models, "artifact", None)
    if artifact is None:
        return {}
    return {
        "data_name": getattr(artifact, "dataset", None),
        "data_is_real": getattr(artifact, "dataset", None) != "synthetic",
        "data_entities": len(getattr(artifact, "entities", []) or []),
        "data_entity_label": getattr(artifact, "entity_label", "store"),
    }
