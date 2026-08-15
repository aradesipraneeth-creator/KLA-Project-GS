"""
KLA Semiconductor Image Restoration — UI Components & Presentation Cards.
Provides modular visual cards, architecture diagrams, and metric summaries.
"""

from typing import Dict, Optional, Any
import numpy as np


def get_architecture_diagram_html() -> str:
    """Returns responsive HTML/CSS architecture diagram for the KLA-HYBRID-V2 model."""
    html = """
    <div style="background-color: #0e1117; border: 1px solid #262730; border-radius: 10px; padding: 20px; margin: 15px 0;">
        <h4 style="color: #58a6ff; margin-top: 0; text-align: center;">⚡ KLA-HYBRID-V2 Architecture Pipeline</h4>
        <div style="display: flex; flex-direction: column; align-items: center; gap: 8px; font-family: monospace; font-size: 13px;">
            
            <div style="background: #1f6feb; color: white; padding: 8px 16px; border-radius: 6px; width: 80%; text-align: center; font-weight: bold;">
                Input Low-Quality Image [1, 1, 128, 128] float32
            </div>
            
            <div style="color: #8b949e;">↓ (Bicubic 2× Residual Branch & Normalized Feature Branch)</div>
            
            <div style="background: #238636; color: white; padding: 8px 16px; border-radius: 6px; width: 80%; text-align: center;">
                Shallow Extraction: 3×3 Conv + Residual Conv Block (base_dim = 32)
            </div>
            
            <div style="color: #8b949e;">↓</div>
            
            <div style="background: #30363d; border: 1px solid #58a6ff; color: #f0f6fc; padding: 12px 16px; border-radius: 8px; width: 88%; text-align: center;">
                <div style="color: #58a6ff; font-weight: bold; margin-bottom: 6px;">6× Hybrid Restoration Groups</div>
                <div style="font-size: 12px; color: #c9d1d9;">
                    Each Group: <b>6 NAF Blocks</b> (DWConv + SimpleGate + SCA) + <b>2 Swin Blocks</b> (W-MSA / SW-MSA) + Local Residual
                </div>
            </div>
            
            <div style="color: #8b949e;">↓</div>
            
            <div style="background: #8957e5; color: white; padding: 8px 16px; border-radius: 6px; width: 80%; text-align: center;">
                Reconstruction Conv + <b>PixelShuffle 2×</b> (128×128 → 256×256)
            </div>
            
            <div style="color: #8b949e;">↓</div>
            
            <div style="background: #da3633; color: white; padding: 8px 16px; border-radius: 6px; width: 80%; text-align: center; font-weight: bold;">
                Global Residual Addition: Bicubic(LQ) + Learned_Residual
            </div>
            
            <div style="color: #8b949e;">↓</div>
            
            <div style="background: #2ea043; color: white; padding: 8px 16px; border-radius: 6px; width: 80%; text-align: center; font-weight: bold; box-shadow: 0 0 10px rgba(46,160,67,0.5);">
                Output Restored High-Resolution Image [1, 1, 256, 256] float32
            </div>
            
        </div>
    </div>
    """
    return html


def format_stats_table(stats_dict: Dict[str, Dict[str, float]]) -> str:
    """Formats image statistics into a clean markdown table."""
    md = "| Image | Min | Max | Mean | Std |\n"
    md += "| :--- | :--- | :--- | :--- | :--- |\n"
    for name, s in stats_dict.items():
        md += f"| **{name}** | `{s['min']:.4f}` | `{s['max']:.4f}` | `{s['mean']:.4f}` | `{s['std']:.4f}` |\n"
    return md


def get_image_stats(arr: np.ndarray) -> Dict[str, float]:
    """Calculates float32 statistics of an array."""
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "shape": f"{arr.shape[0]}×{arr.shape[1]}",
        "dtype": str(arr.dtype),
    }
