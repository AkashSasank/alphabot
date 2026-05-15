"""Visualization utilities for realtime market charting."""

from tradingbot.visualization.constants import IndicatorUnit, IndicatorUnitRegistry

try:
    from tradingbot.visualization.realtime_plot import (
        RealtimeTradingChart,
        run_live_poll_plot,
    )
except ModuleNotFoundError:
    pass

try:
    from tradingbot.visualization.gradio_chart import GradioChartApp
except ModuleNotFoundError:
    pass

try:
    from tradingbot.visualization.html_chart import InteractiveHTMLChart
except ModuleNotFoundError:
    pass

__all__ = [
    "IndicatorUnit",
    "IndicatorUnitRegistry",
]

if "RealtimeTradingChart" in globals():
    __all__.append("RealtimeTradingChart")

if "run_live_poll_plot" in globals():
    __all__.append("run_live_poll_plot")

if "GradioChartApp" in globals():
    __all__.append("GradioChartApp")

if "InteractiveHTMLChart" in globals():
    __all__.append("InteractiveHTMLChart")
