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

`read_sales` will parse any CSV handed to it. It does not evaluate anything and there is
no `pickle`, no `eval`, and no `yaml.load` anywhere in the package, so a hostile CSV can
produce a wrong answer or an exception but not code execution.

Model artefacts, once `models/gbm.py` exists, will be LightGBM's own text format rather
than pickles, for the same reason: loading a pickle from an untrusted source is arbitrary
code execution and there is no need to accept that risk here.

## Dependencies

Pinned exactly in `pyproject.toml`. Dependabot opens weekly grouped pull requests so the
pins do not silently rot, and CodeQL runs on every push to `main` plus weekly, so a newly
published advisory is caught even during a quiet period.

## Reporting

Open an issue at https://github.com/Surge77/stockout/issues. For anything you would rather
not post publicly, email tejasdeshmane66@gmail.com.

This is a personal project with no service attached and no SLA — expect a best-effort
response, not an incident process.
