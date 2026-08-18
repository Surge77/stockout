# Security

## Scope

`stockout` is an offline analysis package. It has no server, no authentication, no user
input beyond command-line flags and CSV paths, and it makes exactly one network call —
`stockout fetch`, which downloads a public dataset from Kaggle over HTTPS.

The realistic risks are correspondingly small, but two are worth naming.

## Credentials

The Kaggle API token is a real secret. It lives in `~/.kaggle/kaggle.json` or in a
gitignored `.env`, and never in the repository.

`.gitignore` covers `.env` and `.env.*` while permitting `.env.example`, which contains
only empty keys and prose. If a token is ever committed, **rotate it at kaggle.com rather
than deleting the line** — a secret pushed to a public repository stays in the git history
after the file is changed, and rotation is the only fix.

## Untrusted input

`read_sales` will parse any CSV handed to it. It does not evaluate anything, and there is
no `eval`, no `exec` and no `yaml.load` on any data path, so a hostile CSV can produce a
wrong answer or an exception but not code execution.

## Model artefacts execute code, and this is the one real risk here

`stockout train` writes a fitted pipeline with `joblib.dump`, and **`joblib.load` is
pickle**: loading an artefact runs whatever code is inside it. An earlier version of this
file said the package contained no pickle anywhere, which stopped being true when
`persistence.py` was added.

A model artefact is therefore exactly as trustworthy as whoever wrote it, and it is not a
format to accept from a stranger. Two things follow, and both are enforced rather than
advised:

- **`persistence.load` refuses any path outside `config.ARTIFACT_DIR`.** The path is not a
  free parameter a caller or a request gets to choose, and `--model-path` is checked
  against that directory before the file is opened.
- **Nothing in this package accepts an uploaded model.** The only artefacts it reads are
  ones it wrote.

That directory check is a guard against a careless caller, **not a sandbox**. It does not
make a malicious `.joblib` safe; it makes one harder to get in front of `load` by accident.
If you need to accept models from elsewhere, convert to a format that does not execute —
ONNX or a plain parameter dump — rather than relying on the path check.

`ARTIFACT_VERSION` is a correctness guard rather than a security one: an old artefact
loaded by new code is refused, because it usually half-works instead of failing.

## Dependencies

Pinned exactly in `pyproject.toml`. Dependabot opens weekly grouped pull requests so the
pins do not silently rot, and CodeQL runs on every push to `main` plus weekly, so a newly
published advisory is caught even during a quiet period.

## Reporting

Open an issue at https://github.com/Surge77/stockout/issues. For anything you would rather
not post publicly, email tejasdeshmane66@gmail.com.

This is a personal project with no service attached and no SLA — expect a best-effort
response, not an incident process.
