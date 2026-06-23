"""Average spectrum visualization functions for DataIngestion.

These plotting functions were extracted from core/data_ingestion.py as part
of the core/visualization layer separation (Public Release v3).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def plot_spectrum(
    x_data: pd.Series,
    y_data: pd.Series,
    x_label: str,
    title: str,
    spectral_region: str,
    output_path: Path,
    dpi: int = 300,
) -> Path:
    """Plot a single spectrum and save to file.

    Args:
        x_data: Spectral axis values (raman shift or wavelength).
        y_data: Intensity values.
        x_label: Label for x-axis.
        title: Figure title.
        spectral_region: Spectral region (R1, R2, R3, R123) — used for x-axis limits.
        output_path: Path to save the PNG file.
        dpi: Output resolution.

    Returns:
        Path to saved file.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(x_data, y_data, color='#1f77b4', linewidth=1.5)
    plt.xlabel(x_label)
    plt.ylabel('Intensity (counts)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if spectral_region == 'R1':
        try:
            x0 = float(x_data.iloc[0] if hasattr(x_data, 'iloc') else x_data[0])
        except Exception:
            x0 = 0.0
        plt.xlim([x0, 4000])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    return output_path
