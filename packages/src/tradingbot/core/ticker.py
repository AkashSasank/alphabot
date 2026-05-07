"""Ticker orchestration for timeframe-specific sequences and indicators."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List

from pydantic import BaseModel
from tradingbot.core.candle_storage import CandleCache, FileCandelStorage
from tradingbot.core.candles import Candle
from tradingbot.core.constants import Interval
from tradingbot.core.indicators import BaseIndicator, IndicatorPoint
from tradingbot.core.protocols import CandleAPIProvider
from tradingbot.core.sequence import Sequence, sequence_builder

LOGGER = logging.getLogger(__name__)


class TickerData(BaseModel):
    """Structured data model for ticker polling and initialization."""

    interval: str
    candles: List[Candle]
    indicators: Dict[str, List[IndicatorPoint]]

    def to_dict(self) -> Dict[str, Any]:
        """Return the full ticker payload as a nested plain dictionary."""
        return self.model_dump(mode="json")


class Ticker:
    """Own candle sequences and indicator values for every supported timeframe."""

    def __init__(
        self,
        name: str,
        indicators: List[BaseIndicator] | None = None,
        candle_api_provider: CandleAPIProvider | None = None,
        candle_cache: CandleCache | None = None,
        limit: int = 25,
        **kwargs: Any,
    ) -> None:
        """Initialize a ticker with one sequence per supported interval."""
        if candle_api_provider is None:
            candle_api_provider = kwargs.get("candle_api")

        self.name = name
        self.interval = Interval.MINUTE
        self.indicators = indicators or []
        self.sequence_builder = sequence_builder
        self.candle_api_provider = candle_api_provider
        self.candle_cache = candle_cache or FileCandelStorage(
            symbol=name,
        )
        self._default_ticker_length = int(limit)
        self._default_sequence_length = int(1.5 * limit)
        self.poll_limit: int | None = limit
        self.sequence_capacity: int | None = self._default_sequence_length
        self.intervals = list(Interval.all())
        self.timeframes = self.intervals
        self.sequences: Dict[str, Sequence] = {
            interval: Sequence(candles=[], interval=interval)
            for interval in self.intervals
        }
        self._indicator_points_by_timeframe: Dict[
            str,
            Dict[str, List[IndicatorPoint]],
        ] = {interval: {} for interval in self.intervals}

        self.init_cache()
        self.init_indicators()

    @property
    def sequence(self) -> Sequence:
        """Return the sequence for the default interval."""
        return self.get_sequence(self.interval)

    @property
    def indicator_values(self) -> Dict[str, List[IndicatorPoint]]:
        """Return indicator values for the default interval."""
        return self.get_all_indicator_values(self.interval)

    def get_sequence(self, timeframe: str | None = None) -> Sequence:
        """Return the sequence for a timeframe, creating it if needed."""
        timeframe = self._normalize_timeframe(timeframe or self.interval)
        self._ensure_interval(timeframe)
        return self.sequences[timeframe]

    def get_all_indicator_values(
        self,
        timeframe: str | None = None,
    ) -> Dict[str, List[IndicatorPoint]]:
        """Return all computed indicator values for one timeframe."""
        timeframe = self._normalize_timeframe(timeframe or self.interval)
        self._ensure_interval(timeframe)
        return self._indicator_points_by_timeframe[timeframe]

    def get_latest_indicator_values(
        self,
        timeframe: str | None = None,
    ) -> Dict[str, IndicatorPoint | None]:
        """Return the latest indicator value per indicator for one timeframe."""
        return {
            name: points[-1] if points else None
            for name, points in self.get_all_indicator_values(timeframe).items()
        }

    def initialize(
        self,
        candles: List[Candle | Dict[str, Any]] | None = None,
        api_limit: int | None = None,
        recompute: bool = True,
        timeframe: str | None = None,
    ) -> TickerData | Dict[str, TickerData]:
        """Initialize all intervals, or one interval when candle data is provided."""
        if candles is not None and timeframe is None:
            raise ValueError("timeframe is required when initializing from candles.")

        if api_limit is not None:
            self.poll_limit = int(api_limit)
            self.sequence_capacity = max(
                int(api_limit),
                self._default_sequence_length,
            )

        if candles is None and timeframe is None:
            initialized: Dict[str, TickerData] = {}
            for interval in self.intervals:
                loaded_candles = self.init_cache(interval=interval)
                if recompute:
                    self.refresh_indicators(interval=interval)
                initialized[interval] = TickerData(
                    interval=interval,
                    candles=loaded_candles,
                    indicators=self.get_all_indicator_values(interval),
                )
            return initialized

        timeframe = self._normalize_timeframe(timeframe or self.interval)
        self._ensure_interval(timeframe)

        if candles is None:
            loaded_candles = self.init_cache(interval=timeframe)
        else:
            raw_candles = self._raw_candle_dicts(candles)
            self.refresh_cache(raw_candles, timeframe=timeframe)
            loaded_candles = self.get_sequence(timeframe).candles

        if recompute:
            self.refresh_indicators(interval=timeframe)

        return TickerData(
            interval=timeframe,
            candles=loaded_candles,
            indicators=self.get_all_indicator_values(timeframe),
        )

    def init_cache(
        self,
        interval: str | None = None,
    ) -> List[Candle] | Dict[str, List[Candle]]:
        """Initialize cache and sequences for all intervals or one interval."""
        if interval is None:
            return {
                item: self.init_cache(interval=item)
                for item in self.intervals
            }

        if self.candle_api_provider is None:
            return self._load_sequence_from_cache(interval)

        interval = self._normalize_timeframe(interval)
        self._ensure_interval(interval)
        cached_candles = self.candle_cache.load(interval=interval)
        start_date = self._fetch_start_date(cached_candles)

        raw_candles = self.candle_api_provider.fetch_candles(
            symbol=self.name,
            interval=interval,
            from_date=start_date,
            to_date=datetime.now(),
        )
        self.candle_cache.save(raw_candles, interval=interval)
        return self._load_sequence_from_cache(interval)

    def init_indicators(self, interval: str | None = None) -> None:
        """Compute initial indicator values for all intervals or one interval."""
        intervals = self.intervals if interval is None else [interval]
        for item in intervals:
            self.refresh_indicators(interval=item)

    def refresh_cache(
        self,
        raw_candles: List[Dict[str, Any]],
        timeframe: str,
    ) -> List[Candle]:
        """Refresh cache and sequence for only the requested timeframe."""
        timeframe = self._normalize_timeframe(timeframe)
        self._ensure_interval(timeframe)
        self.candle_cache.save(raw_candles, interval=timeframe)
        candles = self.candle_cache.load(interval=timeframe)[
            -self._resolved_sequence_length() :
        ]

        sequence = self.get_sequence(timeframe)
        if sequence.candles:
            sequence.update_sequence(candles)
        else:
            self.sequences[timeframe] = self.sequence_builder.build_sequence(
                candles,
                timeframe,
            )
        return candles

    def refresh_indicators(self, interval: str) -> None:
        """Recompute indicators for one timeframe."""
        interval = self._normalize_timeframe(interval)
        self._ensure_interval(interval)
        sequence = self.get_sequence(interval)
        indicator_points = self._indicator_points_by_timeframe[interval]
        for indicator in self.indicators:
            points = indicator.compute(sequence)
            indicator_points[indicator.name] = points[-self._default_ticker_length :]

    def recompute_indicators(self, timeframe: str) -> None:
        """Backward-compatible alias for refreshing indicators."""
        self.refresh_indicators(interval=timeframe)

    def poll(
        self,
        timeframe: str,
        date: datetime | None = None,
        limit: int | None = None,
    ) -> TickerData:
        """Fetch and refresh candles for only one timeframe."""
        if self.candle_api_provider is None:
            raise ValueError("Candle API provider is required for polling.")

        timeframe = self._normalize_timeframe(timeframe)
        self._ensure_interval(timeframe)
        if limit is not None:
            self.poll_limit = int(limit)

        from_date = self.candle_cache.get_latest_timestamp(interval=timeframe)
        if from_date is None and date is not None:
            from_date = (
                date - self._timeframe_delta(timeframe) * self._resolved_limit()
            )
        print(from_date)
        raw_candles = self.candle_api_provider.fetch_candles(
            symbol=self.name,
            interval=timeframe,
            from_date=from_date,
            to_date=date,
        )
        self.refresh_cache(raw_candles, timeframe=timeframe)
        self.refresh_indicators(interval=timeframe)
        return TickerData(
            interval=timeframe,
            candles=self.get_sequence(timeframe).candles,
            indicators=self.get_all_indicator_values(timeframe),
        )

    def apply_polled_candles(
        self,
        candles: List[Candle | Dict[str, Any]],
        timeframe: str,
    ) -> TickerData:
        """Apply externally fetched candles to one timeframe."""
        timeframe = self._normalize_timeframe(timeframe)
        raw_candles = self._raw_candle_dicts(candles)
        self.refresh_cache(raw_candles, timeframe=timeframe)
        self.refresh_indicators(interval=timeframe)
        return TickerData(
            interval=timeframe,
            candles=self.get_sequence(timeframe).candles,
            indicators=self.get_all_indicator_values(timeframe),
        )

    def _load_sequence_from_cache(self, timeframe: str) -> List[Candle]:
        timeframe = self._normalize_timeframe(timeframe)
        candles = self.candle_cache.load(interval=timeframe)[
            -self._resolved_sequence_length() :
        ]
        self.sequences[timeframe] = self.sequence_builder.build_sequence(
            candles,
            timeframe,
        )
        return candles

    def _ensure_interval(self, interval: str) -> None:
        interval = self._normalize_timeframe(interval)
        if interval not in self.intervals:
            raise ValueError(f"Unsupported interval: {interval}")
        self.sequences.setdefault(interval, Sequence(candles=[], interval=interval))
        self._indicator_points_by_timeframe.setdefault(interval, {})

    def _resolved_limit(self) -> int:
        if self.poll_limit is not None and self.poll_limit > 0:
            return int(self.poll_limit)
        return self._default_ticker_length

    def _resolved_sequence_length(self) -> int:
        if self.sequence_capacity is not None and self.sequence_capacity > 0:
            return int(self.sequence_capacity)
        return self._default_sequence_length

    @staticmethod
    def _fetch_start_date(cached_candles: List[Candle]) -> datetime:
        if not cached_candles:
            return datetime(1996, 1, 1)
        if len(cached_candles) == 1:
            return cached_candles[-1].timestamp
        return cached_candles[-2].timestamp

    @staticmethod
    def _normalize_timeframe(timeframe: str) -> str:
        return Interval.normalize(timeframe)

    @staticmethod
    def _raw_candle_dicts(
        candles: List[Candle | Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        raw_candles: List[Dict[str, Any]] = []
        for candle in candles:
            if isinstance(candle, Candle):
                raw_candles.append(
                    {
                        "timestamp": candle.timestamp,
                        "open": candle.open,
                        "high": candle.high,
                        "low": candle.low,
                        "close": candle.close,
                        "volume": candle.volume,
                    }
                )
            else:
                raw_candles.append(candle)
        return raw_candles

    @staticmethod
    def _timeframe_delta(timeframe: str) -> timedelta:
        minutes = Interval.MINUTES_PER_CANDLE.get(Interval.normalize(timeframe), 1)
        return timedelta(minutes=minutes)
