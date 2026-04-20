"""Protocol contracts shared by ticker and indicator modules."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Protocol

from tradingbot.core.candles import Candle
from tradingbot.core.sequence import IndicatorPoint

if TYPE_CHECKING:
    from tradingbot.core.sequence import Sequence


class CandleAPIProvider(Protocol):
    """External API contract for candle retrieval."""

    def fetch_candles(
        self,
        symbol: str,
        interval: str,
        limit: int,
        as_of: datetime | None = None,
    ) -> List[Candle | Dict[str, Any]]: ...


class Indicator(Protocol):
    """Runtime contract expected by ticker orchestration."""

    name: str
    description: str

    def compute_point(self, candles: list[Candle]) -> IndicatorPoint: ...

    def compute(self, sequence: "Sequence") -> list[IndicatorPoint]: ...
