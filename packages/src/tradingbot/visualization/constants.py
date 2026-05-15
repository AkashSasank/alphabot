"""Constants for visualization unit mapping and grouping."""

from __future__ import annotations

from enum import Enum
from typing import Dict


class IndicatorUnit(str, Enum):
    """Known indicator units used for subplot grouping."""

    PRICE = "price"
    PERCENT = "percent"
    VOLUME = "volume"
    OSCILLATOR = "oscillator"
    VOLATILITY = "volatility"
    RAW = "raw"


class IndicatorUnitRegistry:
    """Central mapping from indicator names to chart units.

    Update this map when adding a new indicator class to control which
    subplot the indicator is rendered on.
    """

    NAME_TO_UNIT: Dict[str, IndicatorUnit] = {
        "MA": IndicatorUnit.PRICE,
        "EMASLOPE": IndicatorUnit.PERCENT,
        "EMA": IndicatorUnit.PRICE,
        "VWAPDIST": IndicatorUnit.PERCENT,
        "VWAP": IndicatorUnit.PRICE,
        "RSI": IndicatorUnit.OSCILLATOR,
        "STOCH": IndicatorUnit.OSCILLATOR,
        "MACD": IndicatorUnit.VOLATILITY,
        "ADX": IndicatorUnit.OSCILLATOR,
        "BBW": IndicatorUnit.PERCENT,
        "ATR": IndicatorUnit.VOLATILITY,
        "RVOL": IndicatorUnit.VOLATILITY,
        "OBV": IndicatorUnit.VOLUME,
    }

    @classmethod
    def unit_for(cls, indicator_name: str) -> IndicatorUnit:
        """Resolve chart unit for indicator by name prefix."""
        normalized = indicator_name.upper().strip()
        for prefix, unit in cls.NAME_TO_UNIT.items():
            if normalized.startswith(prefix):
                return unit
        return IndicatorUnit.RAW

    @classmethod
    def register(cls, indicator_prefix: str, unit: IndicatorUnit) -> None:
        """Register or override unit mapping for custom indicators."""
        cls.NAME_TO_UNIT[indicator_prefix.upper().strip()] = unit
