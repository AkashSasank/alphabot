"""Core trading domain package.

This package contains foundational building blocks used across the trading bot:
- candle models and pattern metadata
- sequence containers and builders
- indicator contracts and implementations
- ticker orchestration for updates and recomputation
"""

from tradingbot.core.candles import Candle, CandleProperties
from tradingbot.core.constants import CandleColor, CandleType, Interval, UpdateOperation
from tradingbot.core.protocols import CandleAPIProvider, Indicator
from tradingbot.core.sequence import IndicatorPoint

__all__ = [
    "Candle",
    "CandleAPIProvider",
    "CandleColor",
    "CandleProperties",
    "CandleType",
    "Indicator",
    "IndicatorPoint",
    "Interval",
    "UpdateOperation",
]
