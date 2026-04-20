"""Candle sequence models and construction helpers.

This module contains the time-series container used by the ticker and indicator
layers, plus a builder utility that normalizes both model objects and
dictionary payloads into ``Sequence`` instances.
"""

import copy
from typing import Any, Dict, List

from pydantic import BaseModel
from tradingbot.core.candles import Candle, CandleBuilder
from tradingbot.core.constants import Interval

__all__ = [
    "IndicatorPoint",
    "Interval",
    "Sequence",
    "SequenceBuilder",
]


class Sequence:
    """Mutable candle container for one symbol and interval."""

    def __init__(
        self,
        candles: List[Candle],
        interval: str,
    ) -> None:
        """Initialize a sequence with candles and interval metadata."""
        self.candles = candles
        self.interval: str = interval

    def append_candle(self, candle: Candle) -> None:
        """Append a new candle to the end of the sequence."""
        self.candles.append(copy.deepcopy(candle))

    def add_candle(self, candle: Candle) -> None:
        """Apply rolling semantics by dropping oldest then appending new.

        If the sequence is empty, this behaves like ``append_candle``.
        """
        if not self.candles:
            self.candles.append(copy.deepcopy(candle))
            return
        self.candles.pop(0)
        self.candles.append(copy.deepcopy(candle))

    def update_candle(self, position: int, candle: Candle) -> None:
        """Replace the candle at a zero-based index with a new candle."""
        self.candles[position] = copy.deepcopy(candle)

    def get_candle(self, position: int) -> Candle:
        """Return the candle stored at the requested zero-based index."""
        return self.candles[position]

    def __repr__(self) -> str:
        """Return a debug-friendly sequence representation."""
        return f"Sequence(candles={self.candles}, interval={self.interval})"


class IndicatorPoint(BaseModel):
    """Single indicator value aligned to one candle timestamp."""

    timestamp: Any
    value: float | None


class SequenceBuilder:
    """Factory for building ``Sequence`` objects from payload variants."""

    def __init__(self):
        """Initialize builders for constructing candle sequences."""
        self.candle_builder = CandleBuilder()
        self.sequence: Sequence = Sequence([], Interval.MINUTE)

    def build_sequence(self, candles: List[Candle], interval: str) -> Sequence:
        """Build a deep-copied sequence from ``Candle`` objects."""
        sequence = copy.deepcopy(self.sequence)
        sequence.candles = [
            self.candle_builder.build_candle(
                c.timestamp, c.open, c.high, c.low, c.close, c.volume
            )
            for c in candles
        ]
        sequence.interval = interval
        return sequence

    def build_sequence_from_dicts(
        self,
        candles: List[Dict[str, Any]],
        interval: str,
    ) -> Sequence:
        """Build a sequence from candle-shaped dictionaries.

        Expected keys per dictionary are timestamp/open/high/low/close/volume.
        """
        sequence = copy.deepcopy(self.sequence)
        sequence.candles = [
            self.candle_builder.build_candle(
                c["timestamp"],
                c["open"],
                c["high"],
                c["low"],
                c["close"],
                c["volume"],
            )
            for c in candles
        ]
        sequence.interval = interval
        return sequence
