"""Core utility functions."""
import os
from pathlib import Path


def require_file(path: Path, description: str = "") -> Path:
    """Validate that a file exists, raising FileNotFoundError if not."""
    path = Path(path)
    if not path.exists():
        msg = f"Required file not found: {path}"
        if description:
            msg = f"{description}: {path}"
        raise FileNotFoundError(msg)
    return path


def resolve_parallel_workers(configured: int, n_items: int) -> int:
    """Determine number of parallel workers for per-point fitting.

    Args:
        configured: User-configured worker count from config.yaml.
            0 = auto (half of available CPU cores).
            1 = sequential (no multiprocessing).
            N > 1 = explicit worker count.
        n_items: Number of work items (points to fit). Workers are
            capped to this value to avoid idle processes.

    Returns:
        Worker count. 1 means run sequentially (skip ProcessPoolExecutor),
        >1 means use parallel execution.
    """
    if configured == 0:
        n_workers = max(1, (os.cpu_count() or 1) // 2)
    else:
        n_workers = max(1, configured)
    return min(n_workers, n_items)


def resolve_trim_proportion(n_points: int, baseline_pct: float = 0.02) -> float:
    """Compute trim proportion that guarantees at least 1 point trimmed per tail.

    scipy.stats.trim_mean computes m = int(proportiontocut * n) points to remove
    from each tail.  With baseline_pct=0.02, scans with fewer than 51 points get
    m=0 (no trimming), defeating cosmic-ray rejection.

    This function returns max(baseline_pct, (1 + 1e-9) / n_points) so that
    int(result * n_points) >= 1 for all n >= 3.  Returns baseline_pct unchanged
    when n <= 2 (scipy gives m=0 → plain mean) or baseline_pct <= 0 (explicit
    no-trim request).

    The epsilon 1e-9 is required for IEEE 754 safety: naive 1/n fails for 82 of
    998 integer values in [3, 1000] (e.g. n=49: int(1/49 * 49) = 0).

    Args:
        n_points: Number of data points being averaged.
        baseline_pct: Per-tail proportion for scipy.stats.trim_mean (default 0.02).

    Returns:
        Adjusted proportion such that int(result * n_points) >= 1 when n >= 3.
    """
    if n_points < 3 or baseline_pct <= 0.0:
        return baseline_pct
    return max(baseline_pct, (1.0 + 1e-9) / n_points)


def format_trim_label(n_points: int, baseline_pct: float = 0.02) -> str:
    """Format trim mean filename token reflecting the effective trim percentage.

    Uses resolve_trim_proportion to compute the effective per-tail proportion,
    then formats as e.g. ``"2p_trim_mean"`` or ``"4p_trim_mean"``.

    Args:
        n_points: Number of points being averaged.
        baseline_pct: Configured per-tail proportion (default 0.02 = 2%).

    Returns:
        Label string such as ``"4p_trim_mean"``.
    """
    effective = resolve_trim_proportion(n_points, baseline_pct)
    pct = round(effective * 100, 1)
    if pct == int(pct):
        return f"{int(pct)}p_trim_mean"
    return f"{pct}p_trim_mean"