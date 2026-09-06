"""Interpretable diabetic-retinopathy grading from fundus photographs.

The package is deliberately split into four independent layers so that each one
can be tested (and reused) on its own:

``dr.data``       dataset adapters, label joining, leakage-aware manifests
``dr.features``   retinal preprocessing and handcrafted biomarker extraction
``dr.modeling``   splits, metrics, estimators, training and evaluation
``dr.cli``        the command line entry point that glues them together
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
