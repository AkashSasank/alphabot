"""Ticker orchestration for sequence lifecycle and indicator computations.

The ``Ticker`` class centralizes three responsibilities:
1) owning and mutating the candle ``Sequence``
2) managing registered indicators
3) recomputing indicator values after sequence changes

It can initialize state from direct candle payloads or from an external candle
API provider and supports update operations for append/rolling/update flows.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List

from pydantic import BaseModel
from tradingbot.core.candle_storage import CandleCache, FileCandelStorage
from tradingbot.core.candles import Candle, CandleBuilder
from tradingbot.core.indicators import BaseIndicator, IndicatorPoint
from tradingbot.core.protocols import CandleAPIProvider, Indicator
from tradingbot.core.sequence import Sequence, sequence_builder

LOGGER = logging.getLogger(__name__)


class TickerData(BaseModel):
    """Structured data model for ticker initialization and updates."""

    interval: str
    candles: List[Candle]
    indicators: Dict[str, List[IndicatorPoint]]

    def to_dict(self) -> Dict[str, Any]:
        """Return the full ticker payload as a nested plain dictionary."""
        return self.model_dump(mode="json")


class Ticker:
    def __init__(
        self,
        name: str,
        interval: str,
        indicators: List[BaseIndicator],
        candle_api_provider: CandleAPIProvider,
        candle_cache: CandleCache,
        limit: int = 25
    ) -> None:
        """Initialize a ticker with symbol, interval, and dependencies.

        Args:
            name: Ticker name (e.g. "AAPL").
            interval: Candle interval (e.g. "1m", "5m", "1d").
            indicators: List of indicators to compute.
            candle_api_provider: Optional API provider for fetching candles.
            candle_cache: Optional cache for persisting candles.
            limit: Maximum number of candles per ticker
        """
        self.name = name
        self.interval = interval
        self.indicators = indicators
        self.sequence_builder = sequence_builder
        self.candle_api_provider = candle_api_provider
        self.candle_cache = candle_cache
        self._default_ticker_length = limit
        self._default_sequence_length = int(1.5*limit)
        self._indicator_points: Dict[str, List[IndicatorPoint]] = {}
        self._sequence: Sequence = Sequence(candles=[], interval=interval)
        self.init_cache()
        self.init_indicators()

    def init_cache(self):
        """Initialize cache with recent candles from API provider, if available."""
        cached_candles = self.candle_cache.load()
        if not cached_candles:
            start_date = datetime(1996, 1, 1)
        else:
            start_date = cached_candles[-2].timestamp
        raw_candles = self.candle_api_provider.fetch_candles(
            symbol=self.name,
            interval=self.interval,
            from_date=start_date,
            to_date=datetime.now(),
        )
        self.candle_cache.save(raw_candles)
        candles = self.candle_cache.load()[-self._default_sequence_length:]
        self._sequence = self.sequence_builder.build_sequence(candles, self.interval)
        return candles

    def init_indicators(self):
        for indicator in self.indicators:
            points = indicator.compute(self._sequence)
            self._indicator_points[indicator.name] = points[
                -self._default_ticker_length:
            ]

    def refresh_cache(self, raw_candles: List[Dict[str, Any]]) -> List[Candle]:
        """Refresh cache with new raw candles and return list of new Candle objects."""
        self.candle_cache.save(raw_candles)
        candles = self.candle_cache.load()[-self._default_sequence_length:]
        self._sequence.update_sequence(candles)
        return candles

    def refresh_indicators(self):
        for indicator in self.indicators:
            points = indicator.compute(self._sequence)
            self._indicator_points[indicator.name] = points[
                -self._default_ticker_length:
            ]

    def poll(self) -> TickerData:
        """Fetch latest candles, update sequence, and recompute indicators."""
        if self.candle_api_provider is None:
            raise ValueError("Candle API provider is required for polling.")
        raw_candles = self.candle_api_provider.fetch_candles(
            symbol=self.name,
            interval=self.interval,
            from_date=self.candle_cache.get_latest_timestamp(),
        )
        self.refresh_cache(raw_candles)
        self.refresh_indicators()
        return TickerData(
            interval=self.interval,
            candles=self._sequence.candles,
            indicators=self._indicator_points,
        )
