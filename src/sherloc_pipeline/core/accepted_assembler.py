from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    return s in ("true", "1", "t", "yes", "y")


def load_per_point_minerals(base: Path, sol: str, target: str, scan: str) -> pd.DataFrame:
    import glob, re
    frames = []
    point_files = sorted(glob.glob(str(base / "minerals_fit" / f"{sol}_{target}_{scan}_R1_point*_fit_peaks.csv")))
    for pth in point_files:
        try:
            df = pd.read_csv(pth)
        except Exception:
            continue
        if df.empty:
            continue
        # Coerce numerics
        for col in ("center_cm1", "fwhm_cm1", "snr"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["center_cm1", "fwhm_cm1", "snr"]) if set(["center_cm1","fwhm_cm1","snr"]).issubset(df.columns) else pd.DataFrame()
        if df.empty:
            continue
        m = re.search(r"_point(\d+)_fit_peaks\.csv$", pth)
        if not m:
            continue
        pidx = int(m.group(1))
        # Build standardized columns
        out = pd.DataFrame({
            "modality": "minerals",
            "point": pidx,
            "mean": df["center_cm1"].astype(float),
            "amplitude": pd.to_numeric(df.get("amplitude_a", df.get("a", 0.0)), errors="coerce").fillna(0.0).astype(float),
            "fwhm": df["fwhm_cm1"].astype(float),
            "snr": df["snr"].astype(float),
            "r_squared": pd.to_numeric(df.get("r2", ""), errors="coerce"),
        })
        if "pass_fwhm" in df.columns:
            out["pass_fwhm"] = df["pass_fwhm"].apply(_to_bool)
        frames.append(out)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def map_mineral_labels(series_cm1: pd.Series) -> pd.Series:
    try:
        from .mineral_id import load_mineral_rules, map_min_id_series
        from sherloc_pipeline.config import get_config
        cfg = get_config()
        rules = load_mineral_rules(
            Path(cfg.fitting.get("library_path")) if isinstance(cfg.fitting, dict) and cfg.fitting.get("library_path") else None,
            inline_rules=cfg.fitting.get("mineral_rules") if isinstance(cfg.fitting, dict) else None,
        )
        s = map_min_id_series(series_cm1, rules)
        # Fill empty via fallback bounds
        mask = s.astype(str).str.strip() == ""
        if mask.any():
            fallback = []
            for v in series_cm1[mask].astype(float).tolist():
                if 980.0 <= v < 1008.0:
                    fallback.append("pyroxene")
                elif 1008.0 <= v <= 1020.0:
                    fallback.append("sulf1_v1")
                else:
                    fallback.append("")
            s.loc[mask] = fallback
        return s
    except Exception:
        # Full fallback
        vals = []
        for v in series_cm1.astype(float).tolist():
            if 980.0 <= v < 1008.0:
                vals.append("pyroxene")
            elif 1008.0 <= v <= 1020.0:
                vals.append("sulf1_v1")
            else:
                vals.append("")
        return pd.Series(vals, index=series_cm1.index)


def classify_minerals(df: pd.DataFrame, f_lo: float = 25.0, f_hi_keep: float = 30.0) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    # Apply SNR gate globally
    df = df[df["snr"] >= 3.0]
    if df.empty:
        return df
    # Keep flag by FWHM band only
    df["keep"] = df["fwhm"] >= f_hi_keep
    # Tentative: in [f_lo, f_hi_keep)
    tent_mask = (df["fwhm"] >= f_lo) & (df["fwhm"] < f_hi_keep)
    df.loc[tent_mask, "keep"] = False
    # Drop below tentative band
    df = df[(df["fwhm"] >= f_lo)]
    if df.empty:
        return df
    # user_keep & reviewed defaults
    df["user_keep"] = df["keep"].astype(bool)
    df["reviewed"] = False
    # peak_ID will be assigned globally later
    df["peak_ID"] = ""
    # Map labels
    df["label_id"] = map_mineral_labels(df["mean"])
    # Order columns
    cols = [
        "modality","point","mean","amplitude","fwhm","snr","r_squared",
        "label_id","peak_ID","keep","user_keep","reviewed","reject_reason",
    ]
    df["reject_reason"] = ""
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols]


def load_organics_accepted(base: Path, sol: str, target: str, scan: str) -> pd.DataFrame:
    p = base / "organics_fit" / f"{sol}_{target}_{scan}_R1_organics_accepted_peaks.csv"
    if not p.exists():
        return pd.DataFrame()
    o = pd.read_csv(p)
    rows = []
    for _, r in o.iterrows():
        rows.append({
            "modality": "organics",
            "point": int(r.get("point")),
            "mean": float(r.get("center_cm1")),
            "amplitude": float(r.get("amplitude_a")),
            "fwhm": float(r.get("fwhm_cm1")),
            "snr": float(r.get("snr")),
            "r_squared": float(r.get("r2")) if "r2" in o.columns else "",
            "label_id": str(r.get("band", "")),
            "peak_ID": "",
            "keep": True,
            "user_keep": True,
            "reviewed": False,
            "reject_reason": "",
        })
    return pd.DataFrame(rows)


def load_hydration_accepted(base: Path, sol: str, target: str, scan: str) -> pd.DataFrame:
    p = base / "hydration_fit" / f"{sol}_{target}_{scan}_R1_hydration_accepted_peaks.csv"
    if not p.exists():
        return pd.DataFrame()
    h = pd.read_csv(p)
    rows = []
    for _, r in h.iterrows():
        rows.append({
            "modality": "hydration",
            "point": int(r.get("point")),
            "mean": float(r.get("center_cm1")),
            "amplitude": float(r.get("amplitude_a")),
            "fwhm": float(r.get("fwhm_cm1")),
            "snr": float(r.get("snr")),
            "r_squared": float(r.get("r2")) if "r2" in h.columns else "",
            "label_id": str(r.get("band", "")),
            "peak_ID": "",
            "keep": True,
            "user_keep": True,
            "reviewed": False,
            "reject_reason": "",
        })
    return pd.DataFrame(rows)


def build_scan_df(base: Path, sol: str, target: str, scan: str, f_lo: float = 25.0, f_hi_keep: float = 30.0) -> pd.DataFrame:
    m_pp = load_per_point_minerals(base=base, sol=sol, target=target, scan=scan)
    m_rows = classify_minerals(m_pp, f_lo=f_lo, f_hi_keep=f_hi_keep)
    # Enrich minerals with per-point r2 from accepted-peaks summary, applied to ALL peaks at that point
    try:
        acc_csv = base / "minerals_fit" / f"{sol}_{target}_{scan}_R1_accepted_peaks.csv"
        if not m_rows.empty and acc_csv.exists():
            acc = pd.read_csv(acc_csv)
            if "point" in acc.columns and "r2" in acc.columns:
                acc = acc.copy()
                acc["point"] = pd.to_numeric(acc["point"], errors="coerce").astype("Int64")
                # Build per-point r2 map (first non-null per point)
                acc_point_r2 = acc.dropna(subset=["r2"]).groupby("point")["r2"].first()
                mx = m_rows.copy()
                mx["point"] = pd.to_numeric(mx["point"], errors="coerce").astype("Int64")
                mx["r_squared"] = mx["r_squared"].where(mx["r_squared"].notna(), mx["point"].map(acc_point_r2))
                m_rows = mx
    except Exception:
        pass
    # Backfill remaining minerals r_squared using per-point AICc summary (model r2), for all points
    try:
        aicc_csv = base / f"{sol}_{target}_{scan}_R1_fit_aicc_summary.csv"
        if not m_rows.empty and aicc_csv.exists():
            aicc = pd.read_csv(aicc_csv)
            if "point" in aicc.columns and "r2" in aicc.columns:
                aicc = aicc.copy()
                aicc["point"] = pd.to_numeric(aicc["point"], errors="coerce").astype("Int64")
                pt_r2 = aicc.set_index("point")["r2"]
                m_rows["point"] = pd.to_numeric(m_rows["point"], errors="coerce").astype("Int64")
                m_rows["r_squared"] = m_rows["r_squared"].where(m_rows["r_squared"].notna(), m_rows["point"].map(pt_r2))
    except Exception:
        pass
    o_rows = load_organics_accepted(base=base, sol=sol, target=target, scan=scan)
    h_rows = load_hydration_accepted(base=base, sol=sol, target=target, scan=scan)
    parts = [x for x in [m_rows, o_rows, h_rows] if x is not None and not x.empty]
    if not parts:
        return pd.DataFrame(columns=[
            "sol","target","scan","modality","point","mean","amplitude","fwhm","snr","r_squared","label_id","peak_ID","keep","user_keep","reviewed","reject_reason"
        ])
    df = pd.concat(parts, ignore_index=True)
    # Deduplicate: prefer keep=True over keep=False when keys match
    key_cols = ["modality","point","label_id"]
    df["_mean_r"] = df["mean"].round(3)
    df["_fwhm_r"] = df["fwhm"].round(3)
    # Sort so that for identical keys the kept (keep=True) rows are ordered first
    df.sort_values(key_cols + ["_mean_r","_fwhm_r","keep"], ascending=[True, True, True, True, True, False], inplace=True)
    df = df.drop_duplicates(subset=key_cols + ["_mean_r","_fwhm_r"], keep="first").drop(columns=["_mean_r","_fwhm_r"]) 
    # Assign global peak_ID across all modalities, sorted by (modality, point, mean)
    df = df.sort_values(["modality","point","mean"]).reset_index(drop=True)
    df["peak_ID"] = [f"f{i+1}" for i in range(len(df))]
    # Attach identifiers
    df.insert(0, "scan", scan)
    df.insert(0, "target", target)
    df.insert(0, "sol", sol)
    return df[[
        "sol","target","scan","modality","point","mean","amplitude","fwhm","snr","r_squared","label_id","peak_ID","keep","user_keep","reviewed","reject_reason"
    ]]


def write_scan_target_project(base: Path, results_root: Path, sol: str, target: str, scan: str, df: pd.DataFrame) -> None:
    # Scan-level
    scan_out = base / f"{sol}_{target}_{scan}_accepted_peaks.csv"
    write_accepted_table(scan_out, df)
    # Target-level
    tgt_out = base.parent / f"{target}_accepted_peaks.csv"
    if tgt_out.exists():
        old = pd.read_csv(tgt_out)
        old = old[~((old["sol"].astype(str)==str(sol)) & (old["target"]==target) & (old["scan"]==scan))]
        df_t = pd.concat([old, df], ignore_index=True)
    else:
        df_t = df
    write_accepted_table(tgt_out, df_t)
    # Project-level
    proj_out = results_root / "SHERLOC_accepted_peaks_master.csv"
    if proj_out.exists():
        oldp = pd.read_csv(proj_out)
        oldp = oldp[~((oldp["sol"].astype(str)==str(sol)) & (oldp["target"]==target) & (oldp["scan"]==scan))]
        df_p = pd.concat([oldp, df], ignore_index=True)
    else:
        df_p = df
    write_accepted_table(proj_out, df_p)


ACCEPTED_PEAKS_SCHEMA_ID = "accepted_peaks"
ACCEPTED_PEAKS_SCHEMA_VERSION = "1.0.0"
ACCEPTED_PEAKS_COLUMNS: List[str] = [
    "sol",
    "target",
    "scan",
    "modality",
    "point",
    "mean",
    "amplitude",
    "fwhm",
    "snr",
    "r_squared",
    "label_id",
    "peak_ID",
    "keep",
    "user_keep",
    "reviewed",
    "reject_reason",
]
ACCEPTED_PEAKS_BOOLEAN_COLUMNS: List[str] = ["keep", "user_keep", "reviewed"]
_SCHEMA_SIDE_CAR_SUFFIX = ".schema.json"


def accepted_peaks_schema() -> Dict[str, Any]:
    return {
        "schema_id": ACCEPTED_PEAKS_SCHEMA_ID,
        "version": ACCEPTED_PEAKS_SCHEMA_VERSION,
        "columns": list(ACCEPTED_PEAKS_COLUMNS),
        "boolean_columns": list(ACCEPTED_PEAKS_BOOLEAN_COLUMNS),
    }


def _schema_sidecar_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(csv_path.suffix + _SCHEMA_SIDE_CAR_SUFFIX)


def write_schema_sidecar(csv_path: Path, *, metadata: Optional[Dict[str, Any]] = None) -> Path:
    sidecar_path = _schema_sidecar_path(csv_path)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    payload = accepted_peaks_schema()
    if metadata:
        payload.update(metadata)
    # Ensure canonical fields remain authoritative
    payload["columns"] = list(ACCEPTED_PEAKS_COLUMNS)
    payload["boolean_columns"] = list(ACCEPTED_PEAKS_BOOLEAN_COLUMNS)
    payload["table"] = csv_path.name
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    with sidecar_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return sidecar_path


def normalize_accepted_peaks_df(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for column in ACCEPTED_PEAKS_COLUMNS:
        if column not in normalized.columns:
            if column in ACCEPTED_PEAKS_BOOLEAN_COLUMNS:
                normalized[column] = False
            elif column == "reject_reason":
                normalized[column] = ""
            else:
                raise ValueError(f"Accepted peaks table missing required column '{column}'")
    normalized = normalized[ACCEPTED_PEAKS_COLUMNS]
    normalized["reject_reason"] = normalized["reject_reason"].fillna("").astype(str)
    for column in ACCEPTED_PEAKS_BOOLEAN_COLUMNS:
        normalized[column] = normalized[column].apply(
            lambda value: False if pd.isna(value) else _to_bool(value)
        )
    return normalized


def write_accepted_table(path: Path, df: pd.DataFrame) -> Path:
    normalized = normalize_accepted_peaks_df(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(path, index=False)
    write_schema_sidecar(path)
    return path

