from __future__ import annotations


def plotly_available() -> bool:
    try:
        import plotly  # noqa: F401

        return True
    except Exception:
        return False

