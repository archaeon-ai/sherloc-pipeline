"""Background-subtraction reference CSVs shipped with the package.

Contents:

- ``Arm_Stowed_post-anomaly_900ppp_trimmed_mean_1266.csv`` — arm-stowed
  dark baseline (post-anomaly Raman calibration; ``bg_type="as"``).
- ``Fused_Silica_Corning7980_Air_Subtracted-Bandwidth-35_SB-Pitt.csv`` —
  fused-silica Pitt-lab reference (``bg_type="fs"``).

These are tier-agnostic algorithmic references — identical bytes serve
both ``PHASE_TIER=team`` and ``PHASE_TIER=public`` deployments. They do
NOT live in R2 (the per-tier ACI / Loupe-workspace contract in
:mod:`web.r2_reader` covers mission data only).
"""
