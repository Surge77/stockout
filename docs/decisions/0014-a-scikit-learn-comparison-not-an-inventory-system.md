# 0014 — A scikit-learn comparison, not an inventory system

## Situation

Up to `v0.4.0` this package ended in a decision layer. It forecast quantiles, calibrated
them conformally, and ran a newsvendor inventory simulator that priced what stocking to
each service level cost. Nine modules, three ADRs of measurement, and a genuinely good
argument: a forecast should be judged by the decision it drives, not by a percentage
error.

Two things were wrong with keeping it.

**It was one model wearing five names.** `gbm`, `gbm_quantile` and `gbm_conformal` were
all LightGBM. The whole visible surface of "which model should I use" was answered by
"the gradient-boosted one", and the interesting comparisons — a penalised linear fit
against an unpenalised one, bagging against a random forest, a kernel against its linear
twin — did not exist because there was nothing to compare.

**The decision layer answered a question the data cannot support.** The cost pair
`Cu = 3, Co = 1` was invented. Every absolute currency figure in the frontier table was
therefore an ordering dressed as a budget, which ADR 0008 admitted and which no amount of
further arithmetic on top of it could fix.

## Decision

The conformal and inventory layer is deleted. `stockout.inventory`, `models/conformal.py`,
`models/conformity.py`, `models/design.py`, `models/gbm.py` and `models/protocols.py` go,
with their nine test files.

What replaces them is a **registry**: twelve regressors and nine classifiers, every one
behind the same `ColumnTransformer`, every one reachable by name from the command line and
from the comparison table. The questions the package now answers are:

| Question | Where it is answered |
|---|---|
| Which model, and did the extra complexity pay? | `evaluate/comparison.py`, registry order — simplest first |
| What does each validation protocol believe about itself? | `evaluate/leakage.py`, four arms against one fixed future |
| Where did each hyperparameter come from? | `models/tuning.py` and `models/grids.py` — ADR 0017 |
| Is today busy *for this shop*? | `targets.py`, per-store terciles — ADR 0016 |
| What does the chosen model say about next Tuesday? | `train.py`, `persistence.py`, `predict.py` |

The parts that were never about inventory are kept unchanged, because they are what makes
any of the above mean anything: the rolling-origin backtest, the three baselines, WMAPE
and MASE, the lag guard, the feature denylist and the property test that perturbs a target
value and asserts no feature moves.

## Cost

**The project's strongest single claim is gone.** "A forecast is judged by the decision it
drives" was the sentence that distinguished this repository from every other forecasting
tutorial, and nothing here replaces it. What is offered instead is breadth with a
measured comparison, which is a weaker claim honestly made.

**ADRs 0007 through 0013 now describe code that does not exist.** They are kept rather
than deleted — a decision record that is edited to match the present is not a record — and
each carries a banner naming this one. A reader following a link from an old commit
message lands somewhere that says so.

**Q5 of `docs/questions.md` became unanswerable.** It asked whether stocking to the
newsvendor quantile beats stocking to the mean, and the simulator that would answer it is
gone. It is rewritten rather than dropped, against a question this package can now
actually settle — see `docs/questions.md`.

**Two years of committed results were invalidated.** Every frontier and coverage number in
the old README and `docs/results.md` was produced by deleted code. They are removed rather
than carried forward as history, because a results file is read as current by default.
