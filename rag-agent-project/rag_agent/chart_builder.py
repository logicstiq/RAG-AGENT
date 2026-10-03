"""Renders a simple bar chart for a grouped aggregation result, as a
base64 PNG — e.g. "average reorder point by category" becomes an actual
chart alongside the exact numbers, not just a table of text.

Deliberately server-side (matplotlib, headless Agg backend) rather than a
JS charting library: no extra script tags to manage in the chat UI, works
identically whether the page is viewed locally or deployed, and the image
is just another field in the JSON response.
"""
from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")  # headless — no display needed, safe on a server
import matplotlib.pyplot as plt


def render_bar_chart(labels: list[str], values: list[float], title: str) -> str:
    """Returns a data: URI (base64 PNG) ready to drop into an <img src>."""
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=130)
    ax.bar(labels, values, color="#3b5bdb")
    ax.set_title(title, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
