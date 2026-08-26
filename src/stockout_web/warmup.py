"""What the app fits *after* it starts answering, rather than before.

Two things are expensive enough to matter and optional enough to defer: the seven
regressors behind the forecast page's "what every model predicted" list, and the admin
comparison tables. Between them they cost thirteen minutes of startup on Rossmann — a
random forest and a voting ensemble over 670,422 rows, plus `svc` — during which the app
answered nothing at all, for two features nobody had opened yet.

Neither is fitted on a sample. That would be faster and dishonest: the forecast page calls
the spread "the honest width of the estimate", and a width widened by sampling noise is
not that. They are fitted in full, just not in the way of the door.

Lives in its own module because both the lifespan and the retrain route need it, and
`main` imports the routers — a helper there would have made the import circular.
"""

from __future__ import annotations

import asyncio

from .service import ModelService


async def warm_service(service: ModelService) -> None:
    """Fit both, in the order the pages are likely to be opened.

    The per-row comparison regressors come first: the forecast page is what most visitors
    open, and the admin tables are behind a role check.

    Every failure is swallowed. A warm-up is an optimisation — the comparison table still
    computes on demand, and a forecast without the spread is exactly the page this app
    already renders for an artefact that has no comparison models at all.
    """
    for label, name in (("model spread", "warm"), ("comparison tables", "comparison_tables")):
        try:
            # Resolved inside the try, not in the loop header. Written as
            # `("model spread", service.warm)` the attribute lookup happened while
            # building the tuple — outside every handler — so a service without the
            # method escaped as an unretrieved task exception rather than the one-line
            # notice this promises.
            await getattr(service, name)()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"{label} warm-up skipped: {exc}")
