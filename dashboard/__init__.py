"""
KLA Semiconductor Image Restoration — Dashboard Module.
"""

from .inference import (
    load_model_from_checkpoint,
    restore_single_image,
    compute_metrics,
    compute_diff_maps,
    compute_fft_magnitude,
    run_batch_test,
)
from .visualization import (
    normalize_for_display,
    array_to_png_bytes,
    array_to_npy_bytes,
    plot_comparison_panels,
    plot_error_heatmap,
    plot_histograms,
    plot_fft_spectrum_comparison,
)
from .components import (
    get_architecture_diagram_html,
    format_stats_table,
    get_image_stats,
)

__all__ = [
    "load_model_from_checkpoint",
    "restore_single_image",
    "compute_metrics",
    "compute_diff_maps",
    "compute_fft_magnitude",
    "run_batch_test",
    "normalize_for_display",
    "array_to_png_bytes",
    "array_to_npy_bytes",
    "plot_comparison_panels",
    "plot_error_heatmap",
    "plot_histograms",
    "plot_fft_spectrum_comparison",
    "get_architecture_diagram_html",
    "format_stats_table",
    "get_image_stats",
]
