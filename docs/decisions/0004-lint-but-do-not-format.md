# 0004 — `ruff check`, never `ruff format`

## Situation

Ruff does two jobs. `check` finds unused imports, shadowed builtins, mutable defaults,
import ordering — real defects. `format` rewrites whitespace to a house style.

## Decision

CI runs `ruff check .` only. `ruff format` is never run, and there is no formatter config.

A formatter's diff is noise on every line it touches, and it makes `git blame` point at
the formatting commit rather than at whoever wrote the logic. On a repository this size,
with one author, the defect-finding half is worth having and the whitespace half is not.

`line-length = 100` is set so that `E501` catches genuinely unreadable lines without
anything rewriting them automatically.

## Cost

**Formatting arguments are possible and nobody arbitrates them.** With `format` in CI,
style questions have a mechanical answer. Without it, consistency depends on attention,
and a second contributor would probably tip the balance the other way.

Carried forward from `expense-analyzer`, which made the same call, so the three projects
in `ML/Projects` stay consistent.
