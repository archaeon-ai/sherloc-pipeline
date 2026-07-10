#!/usr/bin/env python
"""ML despike integration parity harness (hard gate).

Proves that the integrated despike path (``ml_despike.featurize_batch``
→ ``MLCRDetector``) reproduces the validated detector behavior exactly:
**zero symmetric difference** between flag sets on the reference
256-frame golden batch (128 R1 + 64 R2 + 64 R3 real validation frames,
selection seed 20260650), at the frozen taus, against
an in-script, dependency-independent reference implementation of the
validated numeric path.

This gate carries the model's evaluation basis over to the integrated
pipeline: the next integration phase does not start, and the change does
not merge, until it passes. Failure is stop-and-diagnose, never deferral.

Requires resources that CI never sees: the model ONNX artifact
and the (gitignored) frame corpus. All paths are argument-supplied — no
built-in defaults.

Usage:
    python scripts/verify_ml_despike_parity.py \\
        --corpus-dir /path/to/frame_corpus \\
        --artifact /path/to/v1_stageB_v13c.onnx \\
        --out /path/to/parity_mark.json

Exit status: 0 on parity (zero symmetric difference), 1 on any
difference or guard failure. The JSON mark written to ``--out`` carries
the required fields (command line, artifact and checkpoint
digests, runtime/provider identification, host identification,
golden-batch identity, and the result) for reproduction in the
verification record.
"""

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# In-script reference implementation of the validated path. Constants and
# math are transcribed from the model's evaluation basis,
# deliberately NOT imported from sherloc_pipeline.ml_despike — the whole
# point is an independent reproduction to compare the integration against.
# ---------------------------------------------------------------------------

REF_N_CHANNELS = 2148
REF_REGIONS = ("R1", "R2", "R3")
REF_REGION_WINDOWS = {"R1": (52, 575), "R2": (575, 1677), "R3": (1677, 2140)}
REF_TAU = {"R1": 0.29882812500038747, "R2": 0.2656250000008831, "R3": 0.2656250000008831}

#: Golden-batch selection: seed and per-region composition.
SELECTION_SEED = 20260650
SELECTION = {"R1": 128, "R2": 64, "R3": 64}

_REF_ONEHOT = {r: np.eye(3, dtype=np.float32)[i] for i, r in enumerate(REF_REGIONS)}


def ref_featurize(active: np.ndarray, dark: np.ndarray, region: str) -> np.ndarray:
    """Validated 8-plane featurization (independent transcription)."""
    lo, hi = REF_REGION_WINDOWS[region]
    x = np.empty((8, REF_N_CHANNELS), dtype=np.float32)
    for k, plane in enumerate((active, dark)):
        win = plane[lo:hi]
        med = np.median(win)
        mad = np.median(np.abs(win - med))
        scale = 1.4826 * mad + 1.0
        x[k] = (plane - med) / scale
        x[5 + k] = np.log10(scale) / 4.0
    oh = _REF_ONEHOT[region]
    x[2] = oh[0]
    x[3] = oh[1]
    x[4] = oh[2]
    x[7] = np.log10(1.0 + abs(float(np.median(active[lo:hi])))) / 4.0
    return x


def ref_flags(probabilities: np.ndarray, region: str) -> set:
    """Window-relative strict-threshold flags, converted to absolute."""
    lo, hi = REF_REGION_WINDOWS[region]
    relative = np.where(probabilities[lo:hi] > REF_TAU[region])[0]
    return set((relative + lo).tolist())


# ---------------------------------------------------------------------------
# Golden-batch reconstruction (replicates the selection semantics
# exactly: val-split filter, per-shard finite-validity mask on strided
# window samples, accumulation order, and RNG consumption order).
# ---------------------------------------------------------------------------


def reconstruct_golden_batch(corpus_dir: Path):
    """Return (actives, darks, regions, corpus_identity) for the 256 frames."""
    manifest_path = corpus_dir / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    scan_split = {sid: m["split"] for sid, m in manifest["scans"].items()}
    keep_scans = {s for s, sp in scan_split.items() if sp == "val"}
    n_shards = manifest["n_shards"]

    # Pass 1: enumerate valid (shard, row) frames per region, in shard
    # order, over the val-filtered rows — identical to the corpus walk.
    frames = {r: [] for r in REF_REGIONS}
    for sh in range(n_shards):
        z = np.load(corpus_dir / f"shard_{sh:03d}.npz", allow_pickle=False)
        sids = z["scan_id"]
        keep = np.array([s in keep_scans for s in sids])
        idx = np.where(keep)[0]
        for r in REF_REGIONS:
            a = np.ascontiguousarray(z[f"{r}_active"][idx])
            d = np.ascontiguousarray(z[f"{r}_dark"][idx])
            lo, hi = REF_REGION_WINDOWS[r]
            valid = np.isfinite(a[:, lo:hi:64]).all(axis=1) & np.isfinite(
                d[:, lo:hi:64]
            ).all(axis=1)
            for row in np.where(valid)[0]:
                frames[r].append((sh, int(row)))
        del z

    # Selection: one RNG, consumed in canonical region order.
    rng = np.random.default_rng(SELECTION_SEED)
    picks = []  # (region, shard, row) in batch order
    for r in REF_REGIONS:
        chosen = rng.choice(len(frames[r]), size=SELECTION[r], replace=False)
        for i in chosen:
            sh, row = frames[r][int(i)]
            picks.append((r, sh, row))

    # Pass 2: load only the shards the selection touches.
    needed_shards = sorted({sh for _, sh, _ in picks})
    shard_data = {}
    for sh in needed_shards:
        z = np.load(corpus_dir / f"shard_{sh:03d}.npz", allow_pickle=False)
        sids = z["scan_id"]
        keep = np.array([s in keep_scans for s in sids])
        idx = np.where(keep)[0]
        shard_data[sh] = {
            r: (
                np.ascontiguousarray(z[f"{r}_active"][idx]),
                np.ascontiguousarray(z[f"{r}_dark"][idx]),
            )
            for r in REF_REGIONS
        }
        del z

    actives, darks, regions = [], [], []
    for r, sh, row in picks:
        a, d = shard_data[sh][r]
        actives.append(a[row])
        darks.append(d[row])
        regions.append(r)

    identity = {
        "selection_seed": SELECTION_SEED,
        "composition": dict(SELECTION),
        "n_frames": len(picks),
        "corpus_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "n_val_frames_per_region": {r: len(frames[r]) for r in REF_REGIONS},
    }
    return actives, darks, regions, identity


def cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ML despike integration parity gate (zero symmetric "
        "difference on the reference 256-frame golden batch)."
    )
    parser.add_argument(
        "--corpus-dir",
        required=True,
        type=Path,
        help="frame corpus directory (corpus_manifest.json + shard_*.npz)",
    )
    parser.add_argument(
        "--artifact",
        required=True,
        type=Path,
        help="model ONNX artifact (digest-verified against the manifest)",
    )
    parser.add_argument(
        "--out", required=True, type=Path, help="output path for the JSON mark"
    )
    args = parser.parse_args()

    import onnxruntime as ort

    from sherloc_pipeline.ml_despike import (
        DEFAULT_MANIFEST,
        MLCRDetector,
        featurize_batch,
    )

    # Verify the supplied artifact against the pinned digest
    # BEFORE any ONNX session is created on it, in either path — the
    # explicit-path no-bypass rule applies to the gate itself.
    print("verifying artifact digest against the pinned manifest ...", flush=True)
    artifact_sha256 = hashlib.sha256(args.artifact.read_bytes()).hexdigest()
    if artifact_sha256 != DEFAULT_MANIFEST.sha256:
        print(
            f"ARTIFACT DIGEST MISMATCH: expected {DEFAULT_MANIFEST.sha256}, "
            f"got {artifact_sha256} for {args.artifact}. No session was "
            "created. The parity gate must run against the validated "
            "artifact — stop and diagnose.",
            file=sys.stderr,
        )
        return 1

    print(f"reconstructing golden batch from {args.corpus_dir} ...", flush=True)
    actives, darks, regions, identity = reconstruct_golden_batch(args.corpus_dir)
    composition = {r: regions.count(r) for r in REF_REGIONS}
    assert composition == SELECTION, f"selection composition {composition}"

    # ---- reference path (in-script, validated configuration) -------------
    print("running in-script reference path ...", flush=True)
    x_ref = np.stack(
        [ref_featurize(a, d, r) for a, d, r in zip(actives, darks, regions)]
    ).astype(np.float32)
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 2
    session_options.inter_op_num_threads = 1
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ref_session = ort.InferenceSession(
        str(args.artifact), session_options, providers=["CPUExecutionProvider"]
    )
    input_name = ref_session.get_inputs()[0].name
    logits = ref_session.run(None, {input_name: x_ref})[0]
    probabilities = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    reference_flags = [
        ref_flags(probabilities[i], regions[i]) for i in range(len(regions))
    ]

    # ---- integrated path --------------------------------------------------
    print("running integrated ml_despike path ...", flush=True)
    detector = MLCRDetector(artifact_path=args.artifact)
    integrated_masks = detector.detect(actives, darks, regions)
    integrated_flags = [set(mask.tolist()) for mask in integrated_masks]

    # ---- comparison --------------------------------------------------------
    n_diff = 0
    frames_with_diff = 0
    flags_ref_total = {r: 0 for r in REF_REGIONS}
    flags_int_total = {r: 0 for r in REF_REGIONS}
    for i, region in enumerate(regions):
        diff = reference_flags[i].symmetric_difference(integrated_flags[i])
        n_diff += len(diff)
        frames_with_diff += bool(diff)
        flags_ref_total[region] += len(reference_flags[i])
        flags_int_total[region] += len(integrated_flags[i])

    parity = n_diff == 0
    mark = {
        "artifact": "verify_ml_despike_parity",
        "result": "PASS" if parity else "FAIL",
        "flag_symmetric_diff": n_diff,
        "frames_with_diff": frames_with_diff,
        "flags_reference_per_region": flags_ref_total,
        "flags_integrated_per_region": flags_int_total,
        "golden_batch": identity,
        "command_line": sys.argv,
        "onnx_artifact_sha256": artifact_sha256,
        "pinned_artifact_sha256": DEFAULT_MANIFEST.sha256,
        "source_checkpoint_sha256": DEFAULT_MANIFEST.checkpoint_sha256,
        "taus": dict(DEFAULT_MANIFEST.tau),
        "onnxruntime_version": ort.__version__,
        "execution_providers": detector._session.get_providers(),
        "session_threads": {
            "intra_op": detector.intra_op_threads,
            "inter_op": detector.inter_op_threads,
        },
        # Diagnostic: the integrated featurization reproduces the
        # reference bit-for-bit (flag parity is the gate; this localizes
        # any failure).
        "featurization_bit_identical": bool(
            np.array_equal(x_ref, featurize_batch(actives, darks, regions))
        ),
        "host": {
            "node": platform.node(),
            "cpu": cpu_name(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(mark, indent=1))

    print(json.dumps({k: mark[k] for k in (
        "result", "flag_symmetric_diff", "frames_with_diff",
        "flags_reference_per_region", "flags_integrated_per_region",
        "onnxruntime_version", "featurization_bit_identical")}, indent=1))
    print(f"mark written to {args.out}")
    if not parity:
        print(
            "PARITY FAILURE: stop and diagnose. The next "
            "integration phase must not start and the change must not merge.",
            file=sys.stderr,
        )
    return 0 if parity else 1


if __name__ == "__main__":
    sys.exit(main())
