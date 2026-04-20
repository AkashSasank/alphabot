from __future__ import annotations

"""Candle helpers and backward-compatible model exports."""

from typing import Any

from pydantic import BaseModel, Field
from tradingbot.core.constants import CandleColor, CandleType

__all__ = [
    "Candle",
    "CandleBuilder",
    "CandleColor",
    "CandleProperties",
    "CandleType",
]


class CandleBuilder:
    """Factory utility to create ``Candle`` instances from raw values."""

    def build_candle(
        self,
        timestamp: Any,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> Candle:
        """Construct and return a normalized ``Candle`` object."""
        return Candle(
            timestamp=timestamp,
            open=open,
            high=high,
            low=low,
            close=close,
            volume=volume,
        )


class CandleProperties(BaseModel):
    """Derived candle metrics used for pattern classification and analysis."""

    candle_type: CandleType = CandleType.STANDARD
    candle_color: CandleColor = CandleColor.NEUTRAL
    body_size: float = 0.0
    upper_wick_size: float = 0.0
    lower_wick_size: float = 0.0
    total_wick_size: float = 0.0
    range_size: float = 0.0
    upper_to_lower_wick_ratio: float | None = None
    wick_to_body_ratio: float | None = None
    body_to_range_ratio: float | None = None


class Candle(BaseModel):
    """Single OHLCV candle with derived structural properties.

    The model computes geometry metrics and inferred pattern metadata in
    ``model_post_init`` so all ``Candle`` instances are analysis-ready.
    """

    timestamp: Any
    open: float
    high: float
    low: float
    close: float
    volume: float
    properties: CandleProperties = Field(default_factory=CandleProperties)

    def model_post_init(self, __context: Any) -> None:
        """Populate derived properties immediately after model creation."""
        self._precalculate_structure_metrics()
        self.properties.candle_color = self._infer_color()
        self.properties.candle_type = self._infer_type()

    @property
    def type(self) -> CandleType:
        """Convenience accessor for inferred candle type."""
        return self.properties.candle_type

    @property
    def color(self) -> CandleColor:
        """Convenience accessor for inferred candle color."""
        return self.properties.candle_color

    @property
    def type_description(self) -> str:
        """Return the description for the inferred candle type."""
        return self.properties.candle_type.description

    def _infer_color(self) -> CandleColor:
        """Infer candle direction from open and close values."""
        if self.close > self.open:
            return CandleColor.GREEN

        if self.close < self.open:
            return CandleColor.RED

        return CandleColor.NEUTRAL

    def _infer_type(self) -> CandleType:
        """Classify candle pattern from body and wick proportions."""
        body = self.properties.body_size
        total_range = self.properties.range_size
        upper_shadow = self.properties.upper_wick_size
        lower_shadow = self.properties.lower_wick_size

        if total_range == 0 or body <= total_range * 0.1:
            return CandleType.DOJI

        if body >= total_range * 0.9:
            return CandleType.MARUBOZU

        if lower_shadow >= body * 2 and upper_shadow <= body:
            return CandleType.HAMMER

        if upper_shadow >= body * 2 and lower_shadow <= body:
            return CandleType.INVERTED_HAMMER

        if body <= total_range * 0.3:
            return CandleType.SPINNING_TOP

        return CandleType.STANDARD

    def _precalculate_structure_metrics(self) -> None:
        """Compute reusable candle geometry metrics and ratios."""
        self.properties.body_size = abs(self.close - self.open)
        self.properties.range_size = max(self.high - self.low, 0.0)
        self.properties.upper_wick_size = max(
            self.high - max(self.open, self.close),
            0.0,
        )
        self.properties.lower_wick_size = max(
            min(self.open, self.close) - self.low,
            0.0,
        )
        self.properties.total_wick_size = (
            self.properties.upper_wick_size + self.properties.lower_wick_size
        )
        self.properties.upper_to_lower_wick_ratio = self._safe_ratio(
            self.properties.upper_wick_size,
            self.properties.lower_wick_size,
        )
        self.properties.wick_to_body_ratio = self._safe_ratio(
            self.properties.total_wick_size,
            self.properties.body_size,
        )
        self.properties.body_to_range_ratio = self._safe_ratio(
            self.properties.body_size,
            self.properties.range_size,
        )

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float) -> float | None:
        """Safely compute a ratio.

        Returns ``None`` when denominator is zero.
        """
        if denominator == 0:
            return None

        return numerator / denominator

    def __repr__(self) -> str:
        """Return a concise debug representation with full candle payload."""
        return (
            "Candle("
            f"timestamp={self.timestamp}, "
            f"open={self.open}, high={self.high}, low={self.low}, "
            f"close={self.close}, volume={self.volume}, "
            f"properties={self.properties}"
            ")"
        )
