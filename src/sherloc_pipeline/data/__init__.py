"""Static algorithmic reference data shipped with the sherloc-pipeline package.

Subpackages here contain calibration / reference inputs that are NOT
per-scan, per-tier mission data (those live in R2 per the m2020-phase
platform spec §3.9). Examples:

- :mod:`sherloc_pipeline.data.background` — arm-stowed + fused-silica
  calibration spectra used by the Workbench background subtraction step
  (``POST /api/process/background``). Tier-agnostic; identical across
  ``team`` and ``public`` deployments.

Loaded via :mod:`importlib.resources` so the resolver works for editable
installs (``pip install -e .``), wheel installs (production container),
and source checkouts alike.
"""
