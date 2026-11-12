# Minerals labeler

A local labeling tool for mineral-window fit components that two fit rules disagree on. One row is one
**disputed component**: persisted under one rule, with no persisted component of the other rule within
10 cm⁻¹ at the same point. The row shows that point's mineral window (700–1200 cm⁻¹, the normalized
baselined R1 spectrum, not despiked), both fits, the disputed band shaded (center ± FWHM/2), and the
components each fit keeps.

The server, persistence, freeze and binding rules are the hydration cosmic-ray labeler's (`cr_labeler`),
unchanged except where listed below. Standard library only; no installs, external assets or build step.

## Classes (neutral names; one key each)

| key | label | meaning |
|---|---|---|
| 1 | `present` | a mineral band is in the shaded region |
| 2 | `absent` | no band there; the component fits noise or baseline |
| 3 | `spike` | the feature is a spike or few-channel artifact, not a band |
| 4 | `unsure` | cannot tell |

**Back**, the left arrow or Backspace revisits the previous row. **Finish** is enabled only when every row has
a label (the server refuses an incomplete set with 422); it then freezes the session. Stopping partway needs
no Finish: labels save on each click and a restart resumes at the first unlabeled row.

## What is shown, and what is not

- Fit **A** and fit **B**: a seeded coin flip per row decides which rule is which letter. The rule names,
  scan, point, rank and unit ID stay in the manifest and are never served.
- Solid line: the sum of that fit's output components (labeled so in the plot). Dashed: each component the
  fit keeps (SNR ≥ 3, FWHM ≥ 25, sharpness passes; on the `disputed` sheet also point R² > 0, the rule the
  comparison set used). The table lists every output component with center, FWHM, SNR and kept; the row in
  the band is highlighted.
- A cosmic-ray mask pixel inside the band is shown as a **review flag**, never a veto.
- Known limit: where the fitter's merge step removed a component from the output table, the solid line is
  not the optimizer's whole model. The builder flags such a fit when its stored point R² exceeds the drawn
  line's R² by more than 0.03, and the UI then says under that fit that the drawn line is short of it.

## Sheets and manifests

`python -m tools.minerals_labeler.manifest --components C.csv --results RESULTS --out M.json [--sheet S]
[--seed N] [--mask-flags F.csv]`

- `--components`: one row per fitted component, persisted or not, columns
  `scan, point, rule, center, amp, fwhm, snr, pass_sharpness, r2` (extra columns ignored). A point with no
  rows for a rule has zero components under that rule.
- `--results`: the results tree; spectra are read from `<results>/<scan>/*_R1_normalized_baselined.csv`.
- `--mask-flags` (optional): `scan, point, center_cm1, mask_px_in_fwhm`.

| sheet | rules | rows | order |
|---|---|---|---|
| `disputed` (default) | `aicc`, `ftest` | every disputed component | seeded shuffle (default seed 20261112) |
| `gain2` | `aicc`, `aicc_gain2` | only the components the AICc gain threshold of 2 loses; kept means SNR ≥ 3, FWHM ≥ 25, sharpness, with **no** R² bar (ruling: r2 none) | SNR ascending, so SNR 3–4 rows come first |

The manifest header records the sheet, rules, order, seed, item count, the component table's SHA-256,
the persist rule, the match tolerance, the window, the class keys and a `public_check` declaration.
**The manifest holds spectra**: keep it on the local machine, outside any repository or served directory.

## Run (loopback only)

```sh
python -m tools.minerals_labeler.server --manifest M.json --out OUTDIR --labeler NAME
```

Open `http://127.0.0.1:8591/`. Non-loopback hosts are refused unless `--allow-nonloopback` is given.
Requests whose `Host` is not the bound address (or `localhost` / `127.0.0.1` / `[::1]` on a loopback bind)
with the bound port, or whose `Origin` does not match it, get 403; this blocks DNS-rebinding reads.
Use a separate output directory per manifest and labeler. Rows are shown in manifest order. A restart
with a different manifest or labeler is refused.

## Labels file format (as `cr_labeler`)

- `labels.jsonl`: append-only `{"id", "label", "labeler", "time"}`, one per click. `id` is the first 16 hex
  characters of HMAC-SHA256(session key, unit_id). `time` is UTC ISO-8601. A later record supersedes an
  earlier one.
- `labels-final.json`: `{"manifest_sha256", "labeler", "labels": {unit_id: label}}`, latest labels only.
  `unit_id` is `<sheet>:<scan>:p<point>:<rule the component is only in>:<center to 0.01>`. Join it to
  the manifest item for the rule letters, SNR and rank.
- `labels-final.sha256`: SHA-256 of the exact final JSON bytes.
- `key` (mode 600) and `session.json` (manifest hash, labeler, order) are never served.

## Tests

`pytest tests/unit/tools/test_minerals_labeler.py` (synthetic data only).
