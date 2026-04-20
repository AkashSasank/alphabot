"""Realtime charting module for candles and indicators.

This module provides a trading-window style matplotlib visualization:
- green/red candlesticks
- timestamp-based x-axis
- indicator legends
- automatic subplot split by indicator unit
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from time import sleep
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib.dates as mdates  # type: ignore[import-not-found]
import matplotlib.pyplot as plt  # type: ignore[import-not-found]
from matplotlib.axes import Axes  # type: ignore[import-not-found]
from matplotlib.patches import Rectangle  # type: ignore[import-not-found]
from tradingbot.bot import BotManager, BotPollData
from tradingbot.core.candles import Candle
from tradingbot.core.sequence import IndicatorPoint
from tradingbot.visualization.constants import IndicatorUnit, IndicatorUnitRegistry


@dataclass
class _Series:
    name: str
    unit: IndicatorUnit
    points: List[IndicatorPoint]


class RealtimeTradingChart:
    """Realtime matplotlib chart for candles and indicator overlays."""

    def __init__(
        self,
        title: str,
        max_candles: int = 200,
        candle_width_minutes: float = 3.5,
    ) -> None:
        self.title = title
        self.max_candles = max_candles
        self.candle_width_minutes = candle_width_minutes
        self.figure = None
        self._axes_by_unit: Dict[IndicatorUnit, Axes] = {}
        self._main_axis: Axes | None = None
        self._window_shown = False

    def render(self, payload: BotPollData) -> None:
        """Render or refresh chart from one poll payload snapshot."""
        series = self._build_series(payload.indicators)
        candles = list(payload.candles)[-self.max_candles :]
        self._prepare_figure(series)

        if not candles:
            self._draw_no_data_placeholder()
            self._finalize_axes()
            self._show_and_flush()
            return

        self._draw_candles(candles)
        self._draw_indicator_series(series)
        self._finalize_axes()

        self._show_and_flush()

    def _show_and_flush(self) -> None:
        """Display figure and process GUI events for interactive backends."""

        if self.figure is not None and not self._window_shown:
            plt.show(block=False)
            self._window_shown = True

        if self.figure is not None and self.figure.canvas is not None:
            self.figure.canvas.draw_idle()
            self.figure.canvas.flush_events()

        plt.pause(0.001)

    def _draw_no_data_placeholder(self) -> None:
        """Render a helpful message when no candles are available."""
        if self._main_axis is None:
            return

        axis = self._main_axis
        axis.set_title(self.title)
        axis.text(
            0.5,
            0.5,
            "No candle data available for the current poll response",
            ha="center",
            va="center",
            transform=axis.transAxes,
            fontsize=12,
        )
        axis.set_ylabel("Price")
        axis.grid(True, alpha=0.25)

    def show(self, block: bool = True) -> None:
        """Open matplotlib window."""
        if self.figure is not None:
            self.figure.show()
        plt.show(block=block)

    def _prepare_figure(self, series: Sequence[_Series]) -> None:
        units = [item.unit for item in series if item.unit != IndicatorUnit.PRICE]
        unique_units: List[IndicatorUnit] = []
        for unit in units:
            if unit not in unique_units:
                unique_units.append(unit)

        rows = 1 + len(unique_units)

        if self.figure is None:
            self.figure, axes = plt.subplots(
                rows,
                1,
                sharex=True,
                figsize=(14, 8 + (2 * len(unique_units))),
            )
            axes_list: List[Any]
            if isinstance(axes, Axes):
                axes_list = [axes]
            elif isinstance(axes, Iterable):
                axes_list = list(axes)
            else:
                axes_list = [axes]

            self._main_axis = axes_list[0]
            self._axes_by_unit = {IndicatorUnit.PRICE: axes_list[0]}

            for index, unit in enumerate(unique_units, start=1):
                self._axes_by_unit[unit] = axes_list[index]
        else:
            current_rows = len(self.figure.axes)
            if current_rows != rows:
                plt.close(self.figure)
                self.figure = None
                self._axes_by_unit = {}
                self._main_axis = None
                self._window_shown = False
                self._prepare_figure(series)
                return

        for axis in self.figure.axes:
            axis.clear()

    def _draw_candles(self, candles: Sequence[Candle]) -> None:
        if self._main_axis is None:
            return

        axis = self._main_axis
        width_days = self.candle_width_minutes / (24 * 60)

        for candle in candles:
            if not isinstance(candle.timestamp, datetime):
                continue

            x_value = float(mdates.date2num(candle.timestamp))
            is_bullish = candle.close >= candle.open
            color = "#1f9d55" if is_bullish else "#d64545"

            axis.vlines(
                x=x_value,
                ymin=candle.low,
                ymax=candle.high,
                color=color,
                linewidth=1,
                alpha=0.9,
            )

            body_bottom = min(candle.open, candle.close)
            body_height = abs(candle.close - candle.open)
            if body_height == 0:
                body_height = 1e-9

            candle_body = Rectangle(
                (x_value - width_days / 2, body_bottom),
                width_days,
                body_height,
                facecolor=color,
                edgecolor=color,
                linewidth=1,
                alpha=0.9,
            )
            axis.add_patch(candle_body)

        axis.set_title(self.title)
        axis.set_ylabel("Price")
        axis.grid(True, alpha=0.25)

    def _draw_indicator_series(self, series: Sequence[_Series]) -> None:
        for item in series:
            axis = self._axes_by_unit.get(item.unit)
            if axis is None:
                continue

            x_values: List[float] = []
            y_values: List[float] = []

            for point in item.points[-self.max_candles :]:
                if not isinstance(point.timestamp, datetime):
                    continue
                if point.value is None:
                    continue
                x_values.append(float(mdates.date2num(point.timestamp)))
                y_values.append(float(point.value))

            if not x_values:
                continue

            axis.plot(x_values, y_values, linewidth=1.5, label=item.name)

            if item.unit == IndicatorUnit.PRICE:
                axis.set_ylabel("Price")
            else:
                axis.set_ylabel(item.unit.value.title())
            axis.grid(True, alpha=0.25)
            axis.legend(loc="upper left")

    def _finalize_axes(self) -> None:
        if self.figure is None:
            return

        bottom_axis = self.figure.axes[-1]
        formatter = mdates.DateFormatter("%Y-%m-%d %H:%M")
        bottom_axis.xaxis.set_major_formatter(formatter)
        bottom_axis.set_xlabel("Timestamp")
        self.figure.autofmt_xdate()
        self.figure.tight_layout()

    @staticmethod
    def _build_series(
        indicators: Dict[str, List[IndicatorPoint]],
    ) -> List[_Series]:
        series: List[_Series] = []
        for name, points in indicators.items():
            series.append(
                _Series(
                    name=name,
                    unit=IndicatorUnitRegistry.unit_for(name),
                    points=list(points),
                )
            )
        return series


def run_live_poll_plot(
    manager: BotManager,
    bot_id: str,
    title: str | None = None,
    interval_seconds: float = 1.0,
    emit_updates: bool = False,
    max_candles: int = 200,
    limit: int | None = None,
    backtest: bool = False,
    backtest_start_date: datetime | None = None,
    backtest_delay_seconds: float = 2.0,
) -> None:
    """Run a live polling loop and render a trading-window chart.

    This function blocks until the plot window is closed or interrupted.
    """
    if title is None:
        title = f"Realtime Trading View - {bot_id}"

    chart = RealtimeTradingChart(title=title, max_candles=max_candles)
    seeded_date = backtest_start_date

    plt.ion()
    try:
        while True:
            if chart.figure is not None:
                figure_number = getattr(chart.figure, "number", None)
                if figure_number is not None and not plt.fignum_exists(figure_number):
                    break
            payload = manager.poll_bot(
                bot_id=bot_id,
                limit=limit,
                emit_updates=emit_updates,
                date=seeded_date,
                backtest=backtest,
                backtest_delay_seconds=backtest_delay_seconds,
            )
            seeded_date = None
            chart.render(payload)
            if not backtest:
                sleep(interval_seconds)
    except KeyboardInterrupt:
        pass
    finally:
        plt.ioff()
        chart.show(block=True)
