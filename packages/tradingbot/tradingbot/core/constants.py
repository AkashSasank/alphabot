"""Core constants and enum types.

This module centralizes shared constant-style definitions used across the
trading core package.
"""

from enum import Enum, StrEnum


class CandleColor(str, Enum):
    """Directional candle color based on open/close relation."""

    GREEN = "green"
    RED = "red"
    NEUTRAL = "neutral"


class CandleType(str, Enum):
    """Common candlestick pattern labels inferred from candle geometry."""

    STANDARD = "standard"
    DOJI = "doji"
    HAMMER = "hammer"
    INVERTED_HAMMER = "inverted_hammer"
    SPINNING_TOP = "spinning_top"
    MARUBOZU = "marubozu"

    @property
    def description(self) -> str:
        """Return a human-readable explanation for the candle pattern."""
        descriptions = {
            CandleType.STANDARD: (
                "A regular candle with no special reversal " "or indecision pattern."
            ),
            CandleType.DOJI: (
                "A candle with a very small body, often " "showing market indecision."
            ),
            CandleType.HAMMER: (
                "A candle with a small body and long lower shadow "
                "that can signal bullish reversal."
            ),
            CandleType.INVERTED_HAMMER: (
                "A candle with a small body and long upper shadow "
                "that can signal a potential reversal."
            ),
            CandleType.SPINNING_TOP: (
                "A candle with a small body and visible upper and "
                "lower shadows, suggesting indecision."
            ),
            CandleType.MARUBOZU: (
                "A strong momentum candle with a large body " "and very small shadows."
            ),
        }
        return descriptions[self]


class Interval:
    """Common interval identifiers used by sequence and ticker components."""

    MINUTE = "minute"
    DAY = "day"
    THREE_MINUTE = "3minute"
    FIVE_MINUTE = "5minute"
    TEN_MINUTE = "10minute"
    FIFTEEN_MINUTE = "15minute"
    THIRTY_MINUTE = "30minute"
    SIXTY_MINUTE = "60minute"

    MINUTES_PER_CANDLE = {
        MINUTE: 1,
        THREE_MINUTE: 3,
        FIVE_MINUTE: 5,
        TEN_MINUTE: 10,
        FIFTEEN_MINUTE: 15,
        THIRTY_MINUTE: 30,
        SIXTY_MINUTE: 60,
        DAY: 60 * 24,
    }


class UpdateOperation(StrEnum):
    """Supported sequence mutation operations used by ticker updates."""

    APPEND = "append"
    ROLLING_ADD = "rolling_add"
    UPDATE = "update"
