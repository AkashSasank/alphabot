"""Concrete indicator implementations.

Each indicator returns timestamp-aligned ``IndicatorPoint`` values so consumers
can map derived metrics back to source candles. The base implementation offers
common utilities and a default full-series computation strategy.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence as SequenceABC
from dataclasses import dataclass
from math import log
from typing import Any

from pydantic import BaseModel
from tradingbot.core.candles import Candle
from tradingbot.core.sequence import Sequence as CandleSequence


@dataclass(frozen=True)
class CandleView:
    """Lightweight OHLCV view used for raw candle dictionaries."""

    timestamp: Any
    open: float
    high: float
    low: float
    close: float
    volume: float


CandleLike = Candle | CandleView | Mapping[str, Any]
CandleInput = CandleSequence | SequenceABC[CandleLike]


class IndicatorPoint(BaseModel):
    """Single indicator value aligned to one candle timestamp."""

    timestamp: Any
    value: float | None


class IndicatorCursor(Iterator[IndicatorPoint]):
    """Stateful cursor that consumes a shared sequence and stores points by timestamp."""

    def __init__(self, indicator: "BaseIndicator", sequence: CandleSequence) -> None:
        self.indicator = indicator
        self.sequence = sequence
        self._points: dict[Any, IndicatorPoint] = {}

    def __iter__(self) -> "IndicatorCursor":
        return self

    def __next__(self) -> IndicatorPoint:
        """Compute the next point, refreshing the latest point when caught up."""
        if not self.sequence.candles:
            raise StopIteration

        point = self._compute_next_unseen()
        if point is not None:
            return point

        return self._refresh_latest()

    @property
    def points(self) -> dict[Any, IndicatorPoint]:
        """Return timestamp-indexed points after syncing with the sequence."""
        self.sync()
        return self._points

    @property
    def latest(self) -> IndicatorPoint | None:
        """Return the most recently stored indicator point."""
        points = self.points
        if not points:
            return None
        return next(reversed(points.values()))

    @property
    def values(self) -> list[IndicatorPoint]:
        """Return stored points as an insertion-ordered list."""
        return list(self.points.values())

    def sync(self) -> dict[Any, IndicatorPoint]:
        """Consume unseen candles and refresh the latest candle point."""
        if not self.sequence.candles:
            return self._points

        while self._compute_next_unseen() is not None:
            pass

        self._refresh_latest()
        return self._points

    def recompute(self) -> dict[Any, IndicatorPoint]:
        """Rebuild all stored points from the current external sequence."""
        self._points = {
            point.timestamp: point
            for point in self.indicator.compute(self.sequence)
        }
        return self._points

    def _compute_next_unseen(self) -> IndicatorPoint | None:
        """Compute the next sequence point whose timestamp is not stored."""
        for index, candle in enumerate(self.sequence.candles):
            if candle.timestamp in self._points:
                continue

            point = self.indicator.compute_point(self.sequence.candles[: index + 1])
            self._points[point.timestamp] = point
            return point

        return None

    def _refresh_latest(self) -> IndicatorPoint:
        """Recompute and upsert the latest point from the full sequence."""
        point = self.indicator.compute_point(self.sequence.candles)
        self._points[point.timestamp] = point
        return point


class BaseIndicator(ABC):
    """Base class with shared utilities for indicator implementations."""

    name: str
    description: str

    @abstractmethod
    def compute_point(self, candles: CandleInput) -> IndicatorPoint:
        """Compute one point from an input candle window.
            This method lets you compute the latest indicator point from a given candle window.
            Can be used when we have to compute the realtime indicator value for the current forming candle.

        Subclasses must implement this method.
        """
        raise NotImplementedError

    def compute(self, candles: CandleInput) -> list[IndicatorPoint]:
        """Compute a full series by evaluating each prefix candle window.
        This method should be used when we have to recompute the entire indicator series for a sequence.
        Increases time complexity if only the latest candle is updating.
        Use compute point in that case and update the indicator series with the new point.
        """
        candle_window = self._candle_window(candles)
        points: list[IndicatorPoint] = []

        for index in range(len(candle_window)):
            points.append(self.compute_point(candle_window[: index + 1]))

        return points

    def cursor(self, sequence: CandleSequence) -> IndicatorCursor:
        """Return a stateful cursor over a shared external candle sequence."""
        return IndicatorCursor(self, sequence)

    @staticmethod
    def _require_candles(candles: SequenceABC[Candle | CandleView]) -> None:
        """Ensure indicator computations are not executed with empty input."""
        if not candles:
            raise ValueError("candles must not be empty")

    @classmethod
    def _candle_window(cls, candles: CandleInput) -> list[Candle | CandleView]:
        """Return attribute-accessible candles without building full Candle models."""
        if isinstance(candles, CandleSequence):
            source = candles.candles
        else:
            source = candles

        return [cls._candle_view(candle) for candle in source]

    @staticmethod
    def _candle_view(candle: CandleLike) -> Candle | CandleView:
        """Return a candle object exposing timestamp/open/high/low/close/volume."""
        if isinstance(candle, (Candle, CandleView)):
            return candle

        required_keys = ("timestamp", "open", "high", "low", "close", "volume")
        missing_keys = [key for key in required_keys if key not in candle]
        if missing_keys:
            raise KeyError(f"candle is missing required keys: {missing_keys}")

        return CandleView(
            timestamp=candle["timestamp"],
            open=float(candle["open"]),
            high=float(candle["high"]),
            low=float(candle["low"]),
            close=float(candle["close"]),
            volume=float(candle["volume"]),
        )

    @staticmethod
    def _ema_full_series(
        values: list[float],
        period: int,
    ) -> list[float | None]:
        """Return full EMA series with ``None`` until warm-up is complete."""
        ema_values: list[float | None] = [None] * len(values)
        if period <= 0 or len(values) < period:
            return ema_values

        multiplier = 2 / (period + 1)
        ema = sum(values[:period]) / period
        ema_values[period - 1] = ema

        for index in range(period, len(values)):
            ema = ((values[index] - ema) * multiplier) + ema
            ema_values[index] = ema

        return ema_values


class SimpleMovingAverage(BaseIndicator):
    """Simple moving average (SMA) over a fixed lookback period."""

    def __init__(self, period: int) -> None:
        if period <= 0:
            raise ValueError("period must be greater than 0")
        self.period = period
        self.name = f"MA{period}"
        self.description = f"Simple Moving Average over {period} periods"

    def compute_point(self, candles: CandleInput) -> IndicatorPoint:
        """Return latest SMA point for the given candle window."""
        candles = self._candle_window(candles)
        self._require_candles(candles)
        value = None
        closes = [candle.close for candle in candles]
        if len(closes) >= self.period:
            value = sum(closes[-self.period :]) / self.period
        return IndicatorPoint(timestamp=candles[-1].timestamp, value=value)


class ExponentialMovingAverage(BaseIndicator):
    """Exponential moving average (EMA) over a fixed lookback period."""

    def __init__(self, period: int) -> None:
        if period <= 0:
            raise ValueError("period must be greater than 0")
        self.period = period
        self.name = f"EMA{period}"
        self.description = f"Exponential Moving Average over {period} periods"

    def compute_point(self, candles: CandleInput) -> IndicatorPoint:
        """Return latest EMA point for the given candle window."""
        candles = self._candle_window(candles)
        self._require_candles(candles)
        closes = [candle.close for candle in candles]
        value = self._ema_full_series(closes, self.period)[-1]
        return IndicatorPoint(timestamp=candles[-1].timestamp, value=value)


class ExponentialMovingAverageSlope(BaseIndicator):
    """Slope of an EMA over a trailing comparison window."""

    def __init__(
        self,
        period: int = 9,
        slope_period: int = 3,
        normalize: bool = True,
    ) -> None:
        if period <= 0:
            raise ValueError("period must be greater than 0")
        if slope_period <= 0:
            raise ValueError("slope_period must be greater than 0")
        self.period = period
        self.slope_period = slope_period
        self.normalize = normalize
        self.name = f"EMASLOPE({period},{slope_period})"
        self.description = (
            f"EMA slope over {slope_period} periods using EMA{period}"
        )

    def compute_point(self, candles: CandleInput) -> IndicatorPoint:
        """Return latest EMA slope point."""
        candles = self._candle_window(candles)
        self._require_candles(candles)
        return self._points_from_candles(candles)[-1]

    def compute(self, candles: CandleInput) -> list[IndicatorPoint]:
        """Return full EMA slope series."""
        candles = self._candle_window(candles)
        self._require_candles(candles)
        return self._points_from_candles(candles)

    def _points_from_candles(
        self,
        candles: SequenceABC[Candle | CandleView],
    ) -> list[IndicatorPoint]:
        closes = [candle.close for candle in candles]
        ema_values = self._ema_full_series(closes, self.period)
        points: list[IndicatorPoint] = []

        for index, candle in enumerate(candles):
            value = None
            previous_index = index - self.slope_period
            if previous_index >= 0:
                current = ema_values[index]
                previous = ema_values[previous_index]
                if current is not None and previous is not None:
                    value = (current - previous) / self.slope_period
                    if self.normalize:
                        denominator = abs(previous)
                        value = None if denominator == 0 else value / denominator

            points.append(IndicatorPoint(timestamp=candle.timestamp, value=value))

        return points


class VolumeSimpleMovingAverage(BaseIndicator):
    """Simple moving average over volume instead of closing price."""

    def __init__(self, period: int) -> None:
        if period <= 0:
            raise ValueError("period must be greater than 0")
        self.period = period
        self.name = f"VMA{period}"
        self.description = f"Volume Moving Average over {period} periods"

    def compute_point(self, candles: CandleInput) -> IndicatorPoint:
        """Return latest volume SMA point for the given candle window."""
        candles = self._candle_window(candles)
        self._require_candles(candles)
        volume_values = [candle.volume for candle in candles]
        value = None
        if len(volume_values) >= self.period:
            value = sum(volume_values[-self.period :]) / self.period
        return IndicatorPoint(timestamp=candles[-1].timestamp, value=value)


class VolumeExponentialMovingAverage(BaseIndicator):
    """Exponential moving average over volume instead of closing price."""

    def __init__(self, period: int) -> None:
        if period <= 0:
            raise ValueError("period must be greater than 0")
        self.period = period
        self.name = f"VEMA{period}"
        self.description = f"Volume Exponential Moving Average over {period} periods"

    def compute_point(self, candles: CandleInput) -> IndicatorPoint:
        """Return latest volume EMA point for the given candle window."""
        candles = self._candle_window(candles)
        self._require_candles(candles)
        volume_values = [candle.volume for candle in candles]
        value = self._ema_full_series(volume_values, self.period)[-1]
        return IndicatorPoint(timestamp=candles[-1].timestamp, value=value)


class VolumeWeightedAveragePrice(BaseIndicator):
    """Volume-weighted average price over full history or fixed window."""

    def __init__(self, period: int | None = None) -> None:
        if period is not None and period <= 0:
            raise ValueError("period must be greater than 0 when provided")
        self.period = period
        period_text = "all candles" if period is None else f"last {period} candles"
        self.name = "VWAP" if period is None else f"VWAP{period}"
        self.description = f"Volume Weighted Average Price over {period_text}"

    def compute_point(self, candles: CandleInput) -> IndicatorPoint:
        """Return latest VWAP point using configured volume-weighted window."""
        candles = self._candle_window(candles)
        self._require_candles(candles)
        window = candles
        if self.period is not None:
            if len(candles) < self.period:
                return IndicatorPoint(
                    timestamp=candles[-1].timestamp,
                    value=None,
                )
            window = candles[-self.period :]

        total_volume = sum(candle.volume for candle in window)
        if total_volume == 0:
            value = None
        else:
            weighted_price_sum = sum(
                ((candle.high + candle.low + candle.close) / 3) * candle.volume
                for candle in window
            )
            value = weighted_price_sum / total_volume

        return IndicatorPoint(timestamp=candles[-1].timestamp, value=value)


class VolumeWeightedAveragePriceDistance(BaseIndicator):
    """Distance of close price from VWAP over full history or a fixed window."""

    def __init__(self, period: int | None = None, normalize: bool = True) -> None:
        if period is not None and period <= 0:
            raise ValueError("period must be greater than 0 when provided")
        self.period = period
        self.normalize = normalize
        period_text = "all candles" if period is None else f"last {period} candles"
        self.name = "VWAPDIST" if period is None else f"VWAPDIST{period}"
        self.description = f"Close price distance from VWAP over {period_text}"

    def compute_point(self, candles: CandleInput) -> IndicatorPoint:
        """Return latest close-to-VWAP distance."""
        candles = self._candle_window(candles)
        self._require_candles(candles)
        return self._points_from_candles(candles)[-1]

    def compute(self, candles: CandleInput) -> list[IndicatorPoint]:
        """Return full close-to-VWAP distance series."""
        candles = self._candle_window(candles)
        self._require_candles(candles)
        return self._points_from_candles(candles)

    def _points_from_candles(
        self,
        candles: SequenceABC[Candle | CandleView],
    ) -> list[IndicatorPoint]:
        points: list[IndicatorPoint] = []

        for index, candle in enumerate(candles):
            value = None
            start = 0
            if self.period is not None:
                if index + 1 < self.period:
                    points.append(IndicatorPoint(timestamp=candle.timestamp, value=None))
                    continue
                start = index - self.period + 1

            window = candles[start : index + 1]
            total_volume = sum(item.volume for item in window)
            if total_volume != 0:
                weighted_price_sum = sum(
                    ((item.high + item.low + item.close) / 3) * item.volume
                    for item in window
                )
                vwap = weighted_price_sum / total_volume
                value = candle.close - vwap
                if self.normalize:
                    value = None if vwap == 0 else value / vwap

            points.append(IndicatorPoint(timestamp=candle.timestamp, value=value))

        return points


class RelativeStrengthIndex(BaseIndicator):
    """Relative Strength Index (RSI) momentum oscillator."""

    def __init__(self, period: int = 14, normalize: bool = False) -> None:
        if period <= 0:
            raise ValueError("period must be greater than 0")
        self.period = period
        self.normalize = normalize
        self.name = f"RSI{period}"
        self.description = f"Relative Strength Index over {period} periods"

    def compute_point(self, candles: CandleInput) -> IndicatorPoint:
        """Return latest RSI point based on average gains and losses."""
        candles = self._candle_window(candles)
        self._require_candles(candles)
        closes = [candle.close for candle in candles]
        value = None

        if len(closes) >= self.period + 1:
            deltas = [
                closes[index] - closes[index - 1] for index in range(1, len(closes))
            ]
            recent_deltas = deltas[-self.period :]
            gains = [max(delta, 0.0) for delta in recent_deltas]
            losses = [max(-delta, 0.0) for delta in recent_deltas]
            avg_gain = sum(gains) / self.period
            avg_loss = sum(losses) / self.period

            if avg_loss == 0:
                value = 100.0
            else:
                rs = avg_gain / avg_loss
                value = 100 - (100 / (1 + rs))

        value = self._normalize_oscillator_value(value)
        return IndicatorPoint(timestamp=candles[-1].timestamp, value=value)

    def compute(self, candles: CandleInput) -> list[IndicatorPoint]:
        """Return full RSI series with one pass over aligned candle windows."""
        candles = self._candle_window(candles)
        self._require_candles(candles)
        closes = [candle.close for candle in candles]
        points: list[IndicatorPoint] = []

        for end_index, candle in enumerate(candles):
            value = None
            if end_index >= self.period:
                deltas = [
                    closes[index] - closes[index - 1]
                    for index in range(end_index - self.period + 1, end_index + 1)
                ]
                gains = [max(delta, 0.0) for delta in deltas]
                losses = [max(-delta, 0.0) for delta in deltas]
                avg_gain = sum(gains) / self.period
                avg_loss = sum(losses) / self.period

                if avg_loss == 0:
                    value = 100.0
                else:
                    rs = avg_gain / avg_loss
                    value = 100 - (100 / (1 + rs))

            points.append(
                IndicatorPoint(
                    timestamp=candle.timestamp,
                    value=self._normalize_oscillator_value(value),
                )
            )

        return points

    def _normalize_oscillator_value(self, value: float | None) -> float | None:
        """Normalize a 0-100 oscillator around its midpoint to -1..1."""
        if value is None or not self.normalize:
            return value
        return (value - 50) / 50

    @staticmethod
    def compute_dataframe(close: Any, period: int = 14, normalize: bool = False) -> Any:
        """Return RSI as a vectorized pandas Series aligned to ``close``."""
        import numpy as np

        if period <= 0:
            raise ValueError("period must be greater than 0")

        delta = close.astype(float).diff()
        gains = delta.clip(lower=0.0)
        losses = (-delta).clip(lower=0.0)
        avg_gain = gains.rolling(window=period, min_periods=period).mean()
        avg_loss = losses.rolling(window=period, min_periods=period).mean()

        rs = avg_gain.div(avg_loss.where(avg_loss != 0))
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.where(avg_loss != 0, 100.0)
        rsi = rsi.where(avg_gain.notna() & avg_loss.notna())

        if normalize:
            rsi = (rsi - 50) / 50
        return rsi.replace([np.inf, -np.inf], np.nan)


class MovingAverageConvergenceDivergence(BaseIndicator):
    """MACD line computed as fast EMA minus slow EMA."""

    def __init__(self, fast_period: int = 12, slow_period: int = 26) -> None:
        if fast_period <= 0 or slow_period <= 0:
            raise ValueError("fast_period and slow_period must be greater than 0")
        if fast_period >= slow_period:
            raise ValueError("fast_period must be less than slow_period")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.name = f"MACD({fast_period},{slow_period})"
        self.description = "MACD line using fast EMA minus slow EMA"

    def compute_point(self, candles: CandleInput) -> IndicatorPoint:
        """Return latest MACD point from fast and slow EMA differentials."""
        candles = self._candle_window(candles)
        self._require_candles(candles)
        closes = [candle.close for candle in candles]
        fast_ema = self._ema_full_series(closes, self.fast_period)[-1]
        slow_ema = self._ema_full_series(closes, self.slow_period)[-1]

        value = None
        if fast_ema is not None and slow_ema is not None:
            value = fast_ema - slow_ema

        return IndicatorPoint(timestamp=candles[-1].timestamp, value=value)

    def compute(self, candles: CandleInput) -> list[IndicatorPoint]:
        """Return full MACD line series from aligned fast and slow EMAs."""
        candles = self._candle_window(candles)
        self._require_candles(candles)
        closes = [candle.close for candle in candles]
        fast_ema = self._ema_full_series(closes, self.fast_period)
        slow_ema = self._ema_full_series(closes, self.slow_period)
        points: list[IndicatorPoint] = []

        for candle, fast_value, slow_value in zip(candles, fast_ema, slow_ema):
            value = None
            if fast_value is not None and slow_value is not None:
                value = fast_value - slow_value
            points.append(IndicatorPoint(timestamp=candle.timestamp, value=value))

        return points


class MACDHistogram(BaseIndicator):
    """MACD histogram computed as MACD line minus signal EMA."""

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        normalize: bool = False,
    ) -> None:
        if fast_period <= 0 or slow_period <= 0 or signal_period <= 0:
            raise ValueError("periods must be greater than 0")
        if fast_period >= slow_period:
            raise ValueError("fast_period must be less than slow_period")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.normalize = normalize
        self.name = f"MACDHIST({fast_period},{slow_period},{signal_period})"
        self.description = "MACD histogram using MACD line minus signal EMA"

    def compute_point(self, candles: CandleInput) -> IndicatorPoint:
        """Return latest MACD histogram point."""
        candles = self._candle_window(candles)
        self._require_candles(candles)
        return self._points_from_candles(candles)[-1]

    def compute(self, candles: CandleInput) -> list[IndicatorPoint]:
        """Return full MACD histogram series."""
        candles = self._candle_window(candles)
        self._require_candles(candles)
        return self._points_from_candles(candles)

    def _points_from_candles(
        self,
        candles: SequenceABC[Candle | CandleView],
    ) -> list[IndicatorPoint]:
        closes = [candle.close for candle in candles]
        fast_ema = self._ema_full_series(closes, self.fast_period)
        slow_ema = self._ema_full_series(closes, self.slow_period)
        macd_values: list[float | None] = []
        valid_macd_values: list[float] = []
        valid_macd_indices: list[int] = []

        for index, (fast_value, slow_value) in enumerate(zip(fast_ema, slow_ema)):
            value = None
            if fast_value is not None and slow_value is not None:
                value = fast_value - slow_value
                valid_macd_values.append(value)
                valid_macd_indices.append(index)
            macd_values.append(value)

        signal_values = self._ema_full_series(valid_macd_values, self.signal_period)
        signal_by_index: list[float | None] = [None] * len(candles)
        for signal_value, candle_index in zip(signal_values, valid_macd_indices):
            signal_by_index[candle_index] = signal_value

        points: list[IndicatorPoint] = []
        for index, candle in enumerate(candles):
            value = None
            macd_value = macd_values[index]
            signal_value = signal_by_index[index]
            if macd_value is not None and signal_value is not None:
                value = macd_value - signal_value
                if self.normalize:
                    value = None if candle.close == 0 else value / candle.close

            points.append(IndicatorPoint(timestamp=candle.timestamp, value=value))

        return points


class VolumeMovingAverageConvergenceDivergence(BaseIndicator):
    """Volume MACD line computed from fast and slow volume EMAs."""

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
    ) -> None:
        if fast_period <= 0 or slow_period <= 0:
            raise ValueError("fast_period and slow_period must be greater than 0")
        if fast_period >= slow_period:
            raise ValueError("fast_period must be less than slow_period")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.name = f"VMACD({fast_period},{slow_period})"
        self.description = "Volume MACD line using fast and slow volume EMAs"

    def compute_point(self, candles: CandleInput) -> IndicatorPoint:
        """Return latest volume MACD point from fast and slow EMA differentials."""
        candles = self._candle_window(candles)
        self._require_candles(candles)
        volumes = [candle.volume for candle in candles]
        fast_ema = self._ema_full_series(volumes, self.fast_period)[-1]
        slow_ema = self._ema_full_series(volumes, self.slow_period)[-1]

        value = None
        if fast_ema is not None and slow_ema is not None:
            value = fast_ema - slow_ema

        return IndicatorPoint(timestamp=candles[-1].timestamp, value=value)


class BollingerBandWidth(BaseIndicator):
    """Bollinger Band Width (BBW) volatility expansion metric."""

    def __init__(self, period: int = 20, std_multiplier: float = 2.0) -> None:
        if period <= 0:
            raise ValueError("period must be greater than 0")
        if std_multiplier <= 0:
            raise ValueError("std_multiplier must be greater than 0")
        self.period = period
        self.std_multiplier = std_multiplier
        self.name = f"BBW({period},{std_multiplier})"
        self.description = "Bollinger Band Width as volatility expansion indicator"

    def compute_point(self, candles: CandleInput) -> IndicatorPoint:
        """Return latest BBW point normalized by moving-average midpoint."""
        candles = self._candle_window(candles)
        self._require_candles(candles)
        closes = [candle.close for candle in candles]
        value = None

        if len(closes) >= self.period:
            window = closes[-self.period :]
            middle = sum(window) / self.period
            if middle != 0:
                variance = sum((item - middle) ** 2 for item in window) / self.period
                std_dev = variance**0.5
                upper = middle + self.std_multiplier * std_dev
                lower = middle - self.std_multiplier * std_dev
                value = (upper - lower) / middle

        return IndicatorPoint(timestamp=candles[-1].timestamp, value=value)


class AverageTrueRange(BaseIndicator):
    """Average True Range (ATR) volatility indicator."""

    def __init__(self, period: int = 14) -> None:
        if period <= 0:
            raise ValueError("period must be greater than 0")
        self.period = period
        self.name = f"ATR{period}"
        self.description = f"Average True Range over {period} periods"

    def compute_point(self, candles: CandleInput) -> IndicatorPoint:
        """Return latest ATR point from trailing true-range values."""
        candles = self._candle_window(candles)
        self._require_candles(candles)
        value = None

        if len(candles) >= self.period + 1:
            true_ranges = self._true_ranges(candles)
            value = sum(true_ranges[-self.period :]) / self.period

        return IndicatorPoint(timestamp=candles[-1].timestamp, value=value)

    @staticmethod
    def _true_ranges(candles: SequenceABC[Candle | CandleView]) -> list[float]:
        """Compute true range values between consecutive candles."""
        true_ranges: list[float] = []

        for index in range(1, len(candles)):
            current = candles[index]
            previous = candles[index - 1]
            true_ranges.append(
                max(
                    current.high - current.low,
                    abs(current.high - previous.close),
                    abs(current.low - previous.close),
                )
            )

        return true_ranges

    @staticmethod
    def compute_dataframe(
        high: Any,
        low: Any,
        close: Any,
        period: int = 14,
    ) -> Any:
        """Return ATR as a vectorized pandas Series aligned to the input index."""
        import numpy as np
        import pandas as pd

        if period <= 0:
            raise ValueError("period must be greater than 0")

        high = high.astype(float)
        low = low.astype(float)
        previous_close = close.astype(float).shift(1)
        true_range = pd.concat(
            [
                high - low,
                (high - previous_close).abs(),
                (low - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        true_range = true_range.where(previous_close.notna())
        atr = true_range.rolling(window=period, min_periods=period).mean()
        return atr.replace([np.inf, -np.inf], np.nan)


class AverageDirectionalIndex(BaseIndicator):
    """Average Directional Index (ADX) trend-strength indicator."""

    def __init__(self, period: int = 14, normalize: bool = False) -> None:
        if period <= 0:
            raise ValueError("period must be greater than 0")
        self.period = period
        self.normalize = normalize
        self.name = f"ADX{period}"
        self.description = f"Average Directional Index over {period} periods"

    def compute_point(self, candles: CandleInput) -> IndicatorPoint:
        """Return latest ADX point."""
        candles = self._candle_window(candles)
        self._require_candles(candles)
        return self._points_from_candles(candles)[-1]

    def compute(self, candles: CandleInput) -> list[IndicatorPoint]:
        """Return full ADX series."""
        candles = self._candle_window(candles)
        self._require_candles(candles)
        return self._points_from_candles(candles)

    def _points_from_candles(
        self,
        candles: SequenceABC[Candle | CandleView],
    ) -> list[IndicatorPoint]:
        plus_dm, minus_dm, true_ranges = self._directional_components(candles)
        dx_by_component_index: list[float | None] = [None] * len(true_ranges)
        adx_by_candle_index: list[float | None] = [None] * len(candles)
        points: list[IndicatorPoint] = []

        if len(true_ranges) >= self.period:
            smoothed_tr = sum(true_ranges[: self.period])
            smoothed_plus_dm = sum(plus_dm[: self.period])
            smoothed_minus_dm = sum(minus_dm[: self.period])

            for component_index in range(self.period - 1, len(true_ranges)):
                if component_index > self.period - 1:
                    smoothed_tr = (
                        smoothed_tr
                        - (smoothed_tr / self.period)
                        + true_ranges[component_index]
                    )
                    smoothed_plus_dm = (
                        smoothed_plus_dm
                        - (smoothed_plus_dm / self.period)
                        + plus_dm[component_index]
                    )
                    smoothed_minus_dm = (
                        smoothed_minus_dm
                        - (smoothed_minus_dm / self.period)
                        + minus_dm[component_index]
                    )

                if smoothed_tr == 0:
                    continue

                plus_di = 100 * smoothed_plus_dm / smoothed_tr
                minus_di = 100 * smoothed_minus_dm / smoothed_tr
                denominator = plus_di + minus_di
                if denominator != 0:
                    dx_by_component_index[component_index] = (
                        100 * abs(plus_di - minus_di) / denominator
                    )

        dx_window: list[float] = []
        previous_adx = None
        for component_index, dx in enumerate(dx_by_component_index):
            if dx is None:
                continue

            candle_index = component_index + 1
            if previous_adx is None:
                dx_window.append(dx)
                if len(dx_window) < self.period:
                    continue
                previous_adx = sum(dx_window[-self.period :]) / self.period
            else:
                previous_adx = ((previous_adx * (self.period - 1)) + dx) / self.period

            value = previous_adx
            if self.normalize:
                value = value / 100
            adx_by_candle_index[candle_index] = value

        for candle_index, candle in enumerate(candles):
            value = adx_by_candle_index[candle_index]

            points.append(IndicatorPoint(timestamp=candle.timestamp, value=value))

        return points

    @staticmethod
    def _directional_components(
        candles: SequenceABC[Candle | CandleView],
    ) -> tuple[list[float], list[float], list[float]]:
        plus_dm: list[float] = []
        minus_dm: list[float] = []
        true_ranges: list[float] = []

        for index in range(1, len(candles)):
            current = candles[index]
            previous = candles[index - 1]
            up_move = current.high - previous.high
            down_move = previous.low - current.low

            plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0.0)
            minus_dm.append(
                down_move if down_move > up_move and down_move > 0 else 0.0
            )
            true_ranges.append(
                max(
                    current.high - current.low,
                    abs(current.high - previous.close),
                    abs(current.low - previous.close),
                )
            )

        return plus_dm, minus_dm, true_ranges


class RollingVolatility(BaseIndicator):
    """Rolling volatility of log returns."""

    def __init__(self, period: int = 20, annualize_factor: float | None = None) -> None:
        if period <= 0:
            raise ValueError("period must be greater than 0")
        if annualize_factor is not None and annualize_factor <= 0:
            raise ValueError("annualize_factor must be greater than 0 when provided")
        self.period = period
        self.annualize_factor = annualize_factor
        self.name = f"RVOL{period}"
        self.description = f"Rolling log-return volatility over {period} periods"

    def compute_point(self, candles: CandleInput) -> IndicatorPoint:
        """Return latest rolling volatility point."""
        candles = self._candle_window(candles)
        self._require_candles(candles)
        return self._points_from_candles(candles)[-1]

    def compute(self, candles: CandleInput) -> list[IndicatorPoint]:
        """Return full rolling volatility series."""
        candles = self._candle_window(candles)
        self._require_candles(candles)
        return self._points_from_candles(candles)

    def _points_from_candles(
        self,
        candles: SequenceABC[Candle | CandleView],
    ) -> list[IndicatorPoint]:
        log_returns: list[float | None] = []
        points: list[IndicatorPoint] = []

        for index in range(1, len(candles)):
            previous_close = candles[index - 1].close
            current_close = candles[index].close
            if previous_close <= 0 or current_close <= 0:
                log_returns.append(None)
            else:
                log_returns.append(log(current_close / previous_close))

        for candle_index, candle in enumerate(candles):
            value = None
            if candle_index >= self.period:
                window = log_returns[candle_index - self.period : candle_index]
                if all(item is not None for item in window):
                    returns = [item for item in window if item is not None]
                    mean = sum(returns) / self.period
                    variance = sum((item - mean) ** 2 for item in returns) / self.period
                    value = variance**0.5
                    if self.annualize_factor is not None:
                        value *= self.annualize_factor**0.5

            points.append(IndicatorPoint(timestamp=candle.timestamp, value=value))

        return points


class StochasticOscillator(BaseIndicator):
    """Fast stochastic %K oscillator over a fixed lookback window."""

    def __init__(self, period: int = 14, normalize: bool = False) -> None:
        if period <= 0:
            raise ValueError("period must be greater than 0")
        self.period = period
        self.normalize = normalize
        self.name = f"STOCH{period}"
        self.description = f"Stochastic oscillator %K over {period} periods"

    def compute_point(self, candles: CandleInput) -> IndicatorPoint:
        """Return latest stochastic %K point for the configured period."""
        candles = self._candle_window(candles)
        self._require_candles(candles)
        value = None

        if len(candles) >= self.period:
            window = candles[-self.period :]
            lowest_low = min(candle.low for candle in window)
            highest_high = max(candle.high for candle in window)
            denominator = highest_high - lowest_low
            if denominator != 0:
                value = ((candles[-1].close - lowest_low) / denominator) * 100

        value = self._normalize_oscillator_value(value)
        return IndicatorPoint(timestamp=candles[-1].timestamp, value=value)

    def _normalize_oscillator_value(self, value: float | None) -> float | None:
        """Normalize a 0-100 oscillator around its midpoint to -1..1."""
        if value is None or not self.normalize:
            return value
        return (value - 50) / 50

    @staticmethod
    def compute_dataframe(
        high: Any,
        low: Any,
        close: Any,
        period: int = 14,
        normalize: bool = False,
    ) -> Any:
        """Return stochastic %K as a vectorized pandas Series."""
        import numpy as np

        if period <= 0:
            raise ValueError("period must be greater than 0")

        lowest_low = low.astype(float).rolling(window=period, min_periods=period).min()
        highest_high = high.astype(float).rolling(window=period, min_periods=period).max()
        denominator = highest_high - lowest_low
        value = ((close.astype(float) - lowest_low) / denominator.where(denominator != 0)) * 100
        if normalize:
            value = (value - 50) / 50
        return value.replace([np.inf, -np.inf], np.nan)


class StochasticRSI(BaseIndicator):
    """Stochastic RSI oscillator computed from an RSI series."""

    def __init__(
        self,
        rsi_period: int = 14,
        stoch_period: int = 14,
        normalize: bool = False,
    ) -> None:
        if rsi_period <= 0:
            raise ValueError("rsi_period must be greater than 0")
        if stoch_period <= 0:
            raise ValueError("stoch_period must be greater than 0")
        self.rsi_period = rsi_period
        self.stoch_period = stoch_period
        self.normalize = normalize
        self.name = f"STOCHRSI({rsi_period},{stoch_period})"
        self.description = (
            "Stochastic RSI oscillator over RSI "
            f"{rsi_period} and stochastic {stoch_period} periods"
        )

    def compute_point(self, candles: CandleInput) -> IndicatorPoint:
        """Return latest stochastic RSI point."""
        candles = self._candle_window(candles)
        self._require_candles(candles)
        rsi_values = self._rsi_series(candles)
        value = None

        if len(rsi_values) >= self.stoch_period:
            window = rsi_values[-self.stoch_period :]
            lowest_rsi = min(window)
            highest_rsi = max(window)
            denominator = highest_rsi - lowest_rsi
            if denominator != 0:
                value = ((rsi_values[-1] - lowest_rsi) / denominator) * 100

        value = self._normalize_oscillator_value(value)
        return IndicatorPoint(timestamp=candles[-1].timestamp, value=value)

    def compute(self, candles: CandleInput) -> list[IndicatorPoint]:
        """Return full stochastic RSI series with aligned timestamps."""
        candles = self._candle_window(candles)
        self._require_candles(candles)
        closes = [candle.close for candle in candles]
        rsi_values: list[float] = []
        points: list[IndicatorPoint] = []

        for end_index, candle in enumerate(candles):
            if end_index >= self.rsi_period:
                window = closes[end_index - self.rsi_period : end_index + 1]
                rsi_values.append(self._rsi_value(window))

            value = None
            if len(rsi_values) >= self.stoch_period:
                window = rsi_values[-self.stoch_period :]
                lowest_rsi = min(window)
                highest_rsi = max(window)
                denominator = highest_rsi - lowest_rsi
                if denominator != 0:
                    value = ((rsi_values[-1] - lowest_rsi) / denominator) * 100

            points.append(
                IndicatorPoint(
                    timestamp=candle.timestamp,
                    value=self._normalize_oscillator_value(value),
                )
            )

        return points

    def _rsi_series(self, candles: CandleInput) -> list[float]:
        """Compute RSI values for each candle with enough lookback."""
        candles = self._candle_window(candles)
        closes = [candle.close for candle in candles]
        rsi_values: list[float] = []

        for end_index in range(self.rsi_period, len(closes)):
            window = closes[end_index - self.rsi_period : end_index + 1]
            rsi_values.append(self._rsi_value(window))

        return rsi_values

    def _rsi_value(self, closes: list[float]) -> float:
        """Return RSI for a close-price window of ``rsi_period + 1`` values."""
        deltas = [
            closes[index] - closes[index - 1]
            for index in range(1, len(closes))
        ]
        gains = [max(delta, 0.0) for delta in deltas]
        losses = [max(-delta, 0.0) for delta in deltas]
        avg_gain = sum(gains) / self.rsi_period
        avg_loss = sum(losses) / self.rsi_period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _normalize_oscillator_value(self, value: float | None) -> float | None:
        """Normalize a 0-100 oscillator around its midpoint to -1..1."""
        if value is None or not self.normalize:
            return value
        return (value - 50) / 50

    @staticmethod
    def compute_dataframe(
        close: Any,
        rsi_period: int = 14,
        stoch_period: int = 14,
        normalize: bool = False,
    ) -> Any:
        """Return StochRSI as a vectorized pandas Series aligned to ``close``."""
        import numpy as np

        if rsi_period <= 0:
            raise ValueError("rsi_period must be greater than 0")
        if stoch_period <= 0:
            raise ValueError("stoch_period must be greater than 0")

        rsi = RelativeStrengthIndex.compute_dataframe(
            close,
            period=rsi_period,
            normalize=False,
        )
        lowest_rsi = rsi.rolling(window=stoch_period, min_periods=stoch_period).min()
        highest_rsi = rsi.rolling(window=stoch_period, min_periods=stoch_period).max()
        denominator = highest_rsi - lowest_rsi
        value = ((rsi - lowest_rsi) / denominator.where(denominator != 0)) * 100
        if normalize:
            value = (value - 50) / 50
        return value.replace([np.inf, -np.inf], np.nan)


class OnBalanceVolume(BaseIndicator):
    """On Balance Volume (OBV) cumulative volume-momentum metric."""

    def __init__(self) -> None:
        self.name = "OBV"
        self.description = "On Balance Volume cumulative momentum indicator"

    def compute_point(self, candles: CandleInput) -> IndicatorPoint:
        """Return latest OBV point from cumulative signed-volume changes."""
        candles = self._candle_window(candles)
        self._require_candles(candles)
        value = None

        if len(candles) >= 2:
            value = 0.0
            for index in range(1, len(candles)):
                value += self._obv_delta(candles[index - 1], candles[index])

        return IndicatorPoint(timestamp=candles[-1].timestamp, value=value)

    @staticmethod
    def _obv_delta(
        previous_candle: Candle | CandleView,
        current_candle: Candle | CandleView,
    ) -> float:
        """Return signed volume contribution between two adjacent candles."""
        if current_candle.close > previous_candle.close:
            return current_candle.volume
        if current_candle.close < previous_candle.close:
            return -current_candle.volume
        return 0.0


class FastStochasticOscillator(StochasticOscillator):
    """Alias class kept for naming compatibility with existing integrations."""

    pass


def build_popular_indicators() -> list[BaseIndicator]:
    """Return a commonly used default indicator bundle for quick setup."""
    return [
        SimpleMovingAverage(period=20),
        SimpleMovingAverage(period=50),
        ExponentialMovingAverage(period=9),
        ExponentialMovingAverage(period=21),
        ExponentialMovingAverageSlope(period=9, slope_period=3),
        VolumeWeightedAveragePrice(),
        VolumeWeightedAveragePriceDistance(period=20),
        RelativeStrengthIndex(period=14),
        MovingAverageConvergenceDivergence(
            fast_period=12,
            slow_period=26,
        ),
        MACDHistogram(
            fast_period=12,
            slow_period=26,
            signal_period=9,
        ),
        BollingerBandWidth(period=20, std_multiplier=2.0),
        AverageTrueRange(period=14),
        AverageDirectionalIndex(period=14),
        RollingVolatility(period=20),
        StochasticOscillator(period=14),
        StochasticRSI(rsi_period=14, stoch_period=14),
        OnBalanceVolume(),
    ]
