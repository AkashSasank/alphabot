"""Protocol contracts shared by ticker and indicator modules."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Protocol

from pydantic import BaseModel
from tradingbot.core.candles import Candle

if TYPE_CHECKING:
    from tradingbot.core.sequence import Sequence


class CandleAPIProvider(Protocol):
    """External API contract for candle retrieval."""

    def fetch_candles(
        self,
        symbol: str,
        interval: str,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
    ) -> List[Dict[str, Any]]: ...


class IndicatorPoint(BaseModel):
    """Single indicator value aligned to one candle timestamp."""

    timestamp: Any
    value: float | None


class Indicator(Protocol):
    """Runtime contract expected by ticker orchestration."""

    name: str
    description: str

    def compute_point(self, candles: list[Candle]) -> IndicatorPoint: ...

    def compute(self, sequence: "Sequence") -> list[IndicatorPoint]: ...
