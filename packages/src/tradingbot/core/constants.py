"""Core constants and enum types.

This module centralizes shared constant-style definitions used across the
trading core package.
"""

from enum import Enum

try:
    from enum import StrEnum
except ImportError:
    class StrEnum(str, Enum):
        """Compatibility fallback for Python versions before 3.11."""


class CandleColor(str, Enum):
    """Binary candle pressure color inferred from wick structure."""

    GREEN = "green"
    RED = "red"

    def one_hot_encode(self) -> dict[str, int]:
        """Return a one-hot encoding dictionary for the candle color."""
        return {
            f"candle_color_{color.value}": int(color == self)
            for color in CandleColor
        }


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

    def one_hot_encode(self) -> dict[str, int]:
        """Return a one-hot encoding dictionary for the candle type."""
        return {
            f"candle_type_{ctype.value}": int(ctype == self) for ctype in CandleType
        }


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

    @classmethod
    def all(cls) -> tuple[str, ...]:
        """Return all supported interval identifiers."""
        return tuple(cls.MINUTES_PER_CANDLE.keys())

    @classmethod
    def normalize(cls, interval: str) -> str:
        """Normalize common interval aliases to supported identifiers."""
        normalized = interval.strip().lower()
        aliases = {
            "1m": cls.MINUTE,
            "1min": cls.MINUTE,
            "1minute": cls.MINUTE,
            "3m": cls.THREE_MINUTE,
            "3min": cls.THREE_MINUTE,
            "5m": cls.FIVE_MINUTE,
            "5min": cls.FIVE_MINUTE,
            "10m": cls.TEN_MINUTE,
            "10min": cls.TEN_MINUTE,
            "15m": cls.FIFTEEN_MINUTE,
            "15min": cls.FIFTEEN_MINUTE,
            "30m": cls.THIRTY_MINUTE,
            "30min": cls.THIRTY_MINUTE,
            "60m": cls.SIXTY_MINUTE,
            "60min": cls.SIXTY_MINUTE,
            "1h": cls.SIXTY_MINUTE,
            "day": cls.DAY,
            "1d": cls.DAY,
        }
        return aliases.get(normalized, normalized)


class UpdateOperation(StrEnum):
    """Supported sequence mutation operations used by ticker updates."""

    APPEND = "append"
    ROLLING_ADD = "rolling_add"
    UPDATE = "update"
