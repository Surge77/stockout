"""Forecasters: three baselines, and the gradient-boosted point and quantile models.

The baselines import cleanly on a bare install. The gradient-boosted models construct on
one too — LightGBM is imported inside `fit`, so listing a model is never the same thing
as requiring its dependency.
"""

from __future__ import annotations
