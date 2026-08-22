"""The web app: a user module that forecasts, and an admin module that trains.

Separate from `stockout` on purpose. `docs/architecture.md` claims everything between
`data/` and `persistence.py` is pure — same input, same output, no I/O — and a web app is
nothing but I/O. Keeping it in its own package makes the dependency direction
unambiguous: the app imports the library, and the library has never heard of the app.

    pip install -e ".[web]"
    uvicorn stockout_web.main:app --reload

Two audiences, two modules:

**User** — sign in, ask for one store-day, and get *both* answers. Predicted sales from
the regression model, and Low/Medium/High from the classification model, with the cut
points the class was decided by. Both are shown because `stockout` solves two problems on
the same rows and a page showing one would be serving half the model.

**Admin** — everything the user sees, plus the artefact's provenance, a button that
retrains and redeploys, the whole-registry comparison table, and the accounts.
"""
