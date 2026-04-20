"""Concrete indicator implementations.

Each indicator returns timestamp-aligned ``IndicatorPoint`` values so consumers
can map derived metrics back to source candles. The base implementation offers
common utilities and a default full-series computation strategy.
"""

from tradingbot.core.candles import Candle
from tradingbot.core.protocols import Indicator
from tradingbot.core.sequence import IndicatorPoint, Sequence


class BaseIndicator:
    """Base class with shared utilities for indicator implementations."""

    name: str
    description: str

    def compute_point(self, candles: list[Candle]) -> IndicatorPoint:
        """Compute one point from an input candle window.

        Subclasses must implement this method.
        """
        raise NotImplementedError

    def compute(self, sequence: Sequence) -> list[IndicatorPoint]:
        """Compute a full series by evaluating each prefix candle window."""
        points: list[IndicatorPoint] = []

        for index in range(len(sequence.candles)):
            points.append(self.compute_point(sequence.candles[: index + 1]))

        return points

    @staticmethod
    def _require_candles(candles: list[Candle]) -> None:
        """Ensure indicator computations are not executed with empty input."""
        if not candles:
            raise ValueError("candles must not be empty")

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

    def compute_point(self, candles: list[Candle]) -> IndicatorPoint:
        """Return latest SMA point for the given candle window."""
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

    def compute_point(self, candles: list[Candle]) -> IndicatorPoint:
        """Return latest EMA point for the given candle window."""
        self._require_candles(candles)
        closes = [candle.close for candle in candles]
        value = self._ema_full_series(closes, self.period)[-1]
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

    def compute_point(self, candles: list[Candle]) -> IndicatorPoint:
        """Return latest VWAP point using configured volume-weighted window."""
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

    def __init__(self, period: int = 14) -> None:
        if period <= 0:
            raise ValueError("period must be greater than 0")
        self.period = period
        self.name = f"RSI{period}"
        self.description = f"Relative Strength Index over {period} periods"

    def compute_point(self, candles: list[Candle]) -> IndicatorPoint:
        """Return latest RSI point based on average gains and losses."""
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

        return IndicatorPoint(timestamp=candles[-1].timestamp, value=value)


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

    def compute_point(self, candles: list[Candle]) -> IndicatorPoint:
        """Return latest MACD point from fast and slow EMA differentials."""
        self._require_candles(candles)
        closes = [candle.close for candle in candles]
        fast_ema = self._ema_full_series(closes, self.fast_period)[-1]
        slow_ema = self._ema_full_series(closes, self.slow_period)[-1]

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

    def compute_point(self, candles: list[Candle]) -> IndicatorPoint:
        """Return latest BBW point normalized by moving-average midpoint."""
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

    def compute_point(self, candles: list[Candle]) -> IndicatorPoint:
        """Return latest ATR point from trailing true-range values."""
        self._require_candles(candles)
        value = None

        if len(candles) >= self.period + 1:
            true_ranges = self._true_ranges(candles)
            value = sum(true_ranges[-self.period :]) / self.period

        return IndicatorPoint(timestamp=candles[-1].timestamp, value=value)

    @staticmethod
    def _true_ranges(candles: list[Candle]) -> list[float]:
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

    def __init__(self, period: int = 14) -> None:
        if period <= 0:
            raise ValueError("period must be greater than 0")
        self.period = period
        self.name = f"STOCH{period}"
        self.description = f"Stochastic oscillator %K over {period} periods"

    def compute_point(self, candles: list[Candle]) -> IndicatorPoint:
        """Return latest stochastic %K point for the configured period."""
        self._require_candles(candles)
        value = None

        if len(candles) >= self.period:
            window = candles[-self.period :]
            lowest_low = min(candle.low for candle in window)
            highest_high = max(candle.high for candle in window)
            denominator = highest_high - lowest_low
            if denominator != 0:
                value = ((candles[-1].close - lowest_low) / denominator) * 100

        return IndicatorPoint(timestamp=candles[-1].timestamp, value=value)


class OnBalanceVolume(BaseIndicator):
    """On Balance Volume (OBV) cumulative volume-momentum metric."""

    def __init__(self) -> None:
        self.name = "OBV"
        self.description = "On Balance Volume cumulative momentum indicator"

    def compute_point(self, candles: list[Candle]) -> IndicatorPoint:
        """Return latest OBV point from cumulative signed-volume changes."""
        self._require_candles(candles)
        value = None

        if len(candles) >= 2:
            value = 0.0
            for index in range(1, len(candles)):
                value += self._obv_delta(candles[index - 1], candles[index])

        return IndicatorPoint(timestamp=candles[-1].timestamp, value=value)

    @staticmethod
    def _obv_delta(previous_candle: Candle, current_candle: Candle) -> float:
        """Return signed volume contribution between two adjacent candles."""
        if current_candle.close > previous_candle.close:
            return current_candle.volume
        if current_candle.close < previous_candle.close:
            return -current_candle.volume
        return 0.0


class FastStochasticOscillator(StochasticOscillator):
    """Alias class kept for naming compatibility with existing integrations."""

    pass


def build_popular_indicators() -> list[Indicator]:
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
        OnBalanceVolume(),
    ]
