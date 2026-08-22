# Security

## Scope

This package reads two csv files that ship with it, fits scikit-learn models, and writes
one file. It opens no sockets, reads no credentials, and has no authentication surface.
There is nothing here to configure wrongly.

## The one real hazard: model artefacts

`stockout train` writes `artifacts/model.joblib`, and `joblib.load` **executes code in the
file it reads**. An artefact is exactly as trustworthy as whoever wrote it.

`persistence.load` refuses any path outside `artifacts/`. That is a guard against a
careless caller, not a sandbox — it does not make a hostile file safe. Do not load a
`.joblib` you did not produce, and do not accept one over a network.

## Reporting

Open an issue, or email tejasdeshmane66@gmail.com. This is a teaching repository with no
deployment and no users, so there is no embargo process and no severity ladder.
