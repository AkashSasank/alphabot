"""Concrete indicator implementations.

Each indicator returns timestamp-aligned ``IndicatorPoint`` values so consumers
can map derived metrics back to source candles. The base implementation offers
common utilities and a default full-series computation strategy.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence as SequenceABC
from dataclasses import dataclass
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

    def _normalize_oscillator_value(self, value: float | None) -> float | None:
        """Normalize a 0-100 oscillator around its midpoint to -1..1."""
        if value is None or not self.normalize:
            return value
        return (value - 50) / 50


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

    def _rsi_series(self, candles: CandleInput) -> list[float]:
        """Compute RSI values for each candle with enough lookback."""
        candles = self._candle_window(candles)
        closes = [candle.close for candle in candles]
        rsi_values: list[float] = []

        for end_index in range(self.rsi_period, len(closes)):
            window = closes[end_index - self.rsi_period : end_index + 1]
            deltas = [
                window[index] - window[index - 1]
                for index in range(1, len(window))
            ]
            gains = [max(delta, 0.0) for delta in deltas]
            losses = [max(-delta, 0.0) for delta in deltas]
            avg_gain = sum(gains) / self.rsi_period
            avg_loss = sum(losses) / self.rsi_period

            if avg_loss == 0:
                rsi_values.append(100.0)
            else:
                rs = avg_gain / avg_loss
                rsi_values.append(100 - (100 / (1 + rs)))

        return rsi_values

    def _normalize_oscillator_value(self, value: float | None) -> float | None:
        """Normalize a 0-100 oscillator around its midpoint to -1..1."""
        if value is None or not self.normalize:
            return value
        return (value - 50) / 50


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
        VolumeWeightedAveragePrice(),
        RelativeStrengthIndex(period=14),
        MovingAverageConvergenceDivergence(
            fast_period=12,
            slow_period=26,
        ),
        BollingerBandWidth(period=20, std_multiplier=2.0),
        AverageTrueRange(period=14),
        StochasticOscillator(period=14),
        StochasticRSI(rsi_period=14, stoch_period=14),
        OnBalanceVolume(),
    ]
