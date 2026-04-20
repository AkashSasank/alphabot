"""Ticker orchestration for sequence lifecycle and indicator computations.

The ``Ticker`` class centralizes three responsibilities:
1) owning and mutating the candle ``Sequence``
2) managing registered indicators
3) recomputing indicator values after sequence changes

It can initialize state from direct candle payloads or from an external candle
API provider and supports update operations for append/rolling/update flows.
"""

import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List
from typing import Sequence as TypingSequence

from tradingbot.core.candles import Candle, CandleBuilder
from tradingbot.core.constants import UpdateOperation
from tradingbot.core.protocols import CandleAPIProvider, Indicator
from tradingbot.core.sequence import IndicatorPoint, Sequence, SequenceBuilder


class Ticker:
    """Stateful symbol+interval container with indicator orchestration.

    A ticker instance tracks one instrument/timeframe and keeps indicator
    output
    caches synchronized whenever the underlying sequence is changed.
    """

    def __init__(
        self,
        name: str,
        interval: str,
        indicators: List[Indicator] | None = None,
        candle_api: CandleAPIProvider | None = None,
        parallel_compute_threshold: int = 8,
        max_workers: int | None = None,
    ) -> None:
        """Create an empty ticker and optionally register indicators.

        Args:
            name: Symbol/instrument identifier.
            interval: Candle interval identifier.
            indicators: Optional indicator set to register immediately.
            candle_api: Optional external provider for polling/initialization.
            parallel_compute_threshold: Indicator count threshold that enables
                automatic parallel recomputation.
            max_workers: Optional thread-pool worker override.
        """
        self.name = name
        self.interval = interval
        self.sequence = Sequence(candles=[], interval=interval)
        self.sequence_builder = SequenceBuilder()
        self.indicators: List[Indicator] = []
        self.indicator_values: Dict[str, List[IndicatorPoint]] = {}
        self._latest_indicator_values: Dict[str, float | None] = {}
        self.candle_builder = CandleBuilder()
        self.candle_api = candle_api
        self.sequence_capacity: int | None = None
        self.poll_limit: int | None = None
        self.parallel_compute_threshold = parallel_compute_threshold
        self.max_workers = max_workers
        self._stream_client: Any | None = None
        self._stream_instrument_token: int | None = None
        self._stream_mode: str = "quote"

        if indicators:
            self.add_indicators(indicators, recompute=True)

    def set_candle_api(self, candle_api: CandleAPIProvider) -> None:
        """Assign or replace the external candle API provider."""
        self.candle_api = candle_api

    def initialize(
        self,
        candles: TypingSequence[Candle | Dict[str, Any]] | None = None,
        api_limit: int | None = None,
        recompute: bool = True,
    ) -> None:
        """Initialize ticker sequence from direct candles or API payload.

        Args:
            candles: Optional candle objects or candle-shaped dictionaries.
            api_limit: Required when loading from ``candle_api``.
            recompute: Whether to recompute registered indicators afterward.
        """
        if candles is None:
            if self.candle_api is None:
                raise ValueError("candle_api is not configured")
            if api_limit is None:
                raise ValueError("api_limit is required when candles are not provided")
            self.poll_limit = int(api_limit)
            candles = self.candle_api.fetch_candles(
                symbol=self.name,
                interval=self.interval,
                limit=api_limit,
            )

        if not candles:
            self.sequence = Sequence(candles=[], interval=self.interval)
            self.sequence_capacity = 0
            if api_limit is not None:
                self.poll_limit = int(api_limit)
            if recompute:
                self.recompute_indicators()
            return

        first = candles[0]
        if isinstance(first, Candle):
            built = self.sequence_builder.build_sequence(
                candles=[self._to_candle(candle) for candle in candles],
                interval=self.interval,
            )
        else:
            built = self.sequence_builder.build_sequence_from_dicts(
                candles=candles,  # type: ignore[arg-type]
                interval=self.interval,
            )

        self.sequence = built
        self.sequence_capacity = len(self.sequence.candles)
        if api_limit is None:
            self.poll_limit = len(self.sequence.candles)
        else:
            self.poll_limit = int(api_limit)
        if recompute:
            self.recompute_indicators()

    def poll(
        self,
        date: datetime | None = None,
    ) -> None:
        """Poll external API and apply append/update semantics per candle.

        The method compares each fetched candle interval bucket against the
        current sequence tail and chooses one of:
        - ``UPDATE`` for matching interval bucket
        - ``ROLLING_ADD`` when capacity is reached
        - ``APPEND`` for normal extension

        Args:
            date: Optional timestamp treated as poll "now".
                Defaults to ``datetime.now()``.
        """
        if self.candle_api is None:
            raise ValueError("candle_api is not configured")

        fetch_limit = self._resolve_poll_limit()

        poll_date = date if date is not None else datetime.now()

        try:
            payload = self.candle_api.fetch_candles(
                symbol=self.name,
                interval=self.interval,
                limit=fetch_limit,
                as_of=poll_date,
            )
        except TypeError:
            # Backward compatibility for providers that don't support as_of.
            payload = self.candle_api.fetch_candles(
                symbol=self.name,
                interval=self.interval,
                limit=fetch_limit,
            )
        candles = [self._to_candle(item) for item in payload]
        self.apply_polled_candles(candles)

    def apply_polled_candles(
        self,
        candles: TypingSequence[Candle | Dict[str, Any]],
    ) -> None:
        """Apply externally fetched candles using poll update semantics."""
        normalized_candles = [self._to_candle(item) for item in candles]
        if not normalized_candles:
            return

        if not self.sequence.candles:
            self.initialize(candles=normalized_candles, recompute=True)
            return

        for candle in normalized_candles:
            normalized_candle = self._normalize_polled_candle(candle)
            last_candle = self.sequence.candles[-1]

            if self._is_same_interval_bucket(
                normalized_candle.timestamp,
                last_candle.timestamp,
            ):
                self.update_sequence(
                    candle=normalized_candle,
                    operation=UpdateOperation.UPDATE,
                )
            elif self._is_newer_interval_bucket(
                normalized_candle.timestamp,
                last_candle.timestamp,
            ):
                if (
                    self.sequence_capacity is not None
                    and self.sequence_capacity > 0
                    and len(self.sequence.candles) >= self.sequence_capacity
                ):
                    self.update_sequence(
                        candle=normalized_candle,
                        operation=UpdateOperation.ROLLING_ADD,
                    )
                else:
                    self.update_sequence(
                        candle=normalized_candle,
                        operation=UpdateOperation.APPEND,
                    )

    def _resolve_poll_limit(self) -> int:
        """Resolve effective poll fetch size from ticker-owned state."""
        if self.poll_limit is not None and self.poll_limit > 0:
            return self.poll_limit

        if self.sequence_capacity is not None and self.sequence_capacity > 0:
            self.poll_limit = self.sequence_capacity
            return self.sequence_capacity

        sequence_len = len(self.sequence.candles)
        if sequence_len > 0:
            self.poll_limit = sequence_len
            return sequence_len

        raise ValueError(
            "poll_limit is not configured; initialize ticker with api_limit "
            "or candle payload before polling"
        )

    def _normalize_polled_candle(self, candle: Candle) -> Candle:
        """Normalize polled candle timestamp into this interval bucket."""
        if not isinstance(candle.timestamp, datetime):
            return candle

        normalized_timestamp = self._bucket_timestamp(candle.timestamp)
        return self.candle_builder.build_candle(
            timestamp=normalized_timestamp,
            open=candle.open,
            high=candle.high,
            low=candle.low,
            close=candle.close,
            volume=candle.volume,
        )

    def _is_same_interval_bucket(self, left: Any, right: Any) -> bool:
        if isinstance(left, datetime) and isinstance(right, datetime):
            aligned_left, aligned_right = self._align_timestamp_timezones(
                left,
                right,
            )
            left_bucket = self._bucket_timestamp(aligned_left)
            right_bucket = self._bucket_timestamp(aligned_right)
            return left_bucket == right_bucket

        return left == right

    def _is_newer_interval_bucket(self, left: Any, right: Any) -> bool:
        if isinstance(left, datetime) and isinstance(right, datetime):
            aligned_left, aligned_right = self._align_timestamp_timezones(
                left,
                right,
            )
            left_bucket = self._bucket_timestamp(aligned_left)
            right_bucket = self._bucket_timestamp(aligned_right)
            return left_bucket > right_bucket

        return left > right

    def stream(
        self,
        websocket_client: Any,
        instrument_token: int | None = None,
        mode: str = "quote",
        auto_connect: bool = True,
    ) -> None:
        """Start websocket-driven streaming updates for this ticker.

        The stream path updates candles and indicators from live ticks with
        lower latency than polling. It keeps the websocket connection open and
        applies update/append semantics equivalent to ``poll``.

        Args:
            websocket_client: Websocket client with ``set_on_ticks`` and
                ``subscribe`` methods. Optionally may provide
                ``is_connected`` and ``connect``.
            instrument_token: Instrument token for this ticker symbol.
                If omitted, this method attempts to resolve token from
                ``candle_api`` when provider exposes ``_get_instrument_token``.
            mode: Stream mode passed to websocket subscribe/connect.
            auto_connect: When true, connect websocket if not connected.
        """
        if websocket_client is None:
            raise ValueError("websocket_client is required")

        token = self._resolve_stream_instrument_token(instrument_token)

        previous_callback = None
        callbacks = getattr(websocket_client, "_callbacks", None)
        if isinstance(callbacks, dict):
            previous_callback = callbacks.get("ticks")

        def _stream_ticks_handler(ws_or_ticks: Any, ticks: Any = None) -> None:
            resolved_ticks = ws_or_ticks if ticks is None else ticks
            self._process_stream_ticks(
                ticks=resolved_ticks,
                instrument_token=token,
            )

            if callable(previous_callback):
                previous_callback(ws_or_ticks, resolved_ticks)

        websocket_client.set_on_ticks(_stream_ticks_handler)

        already_connected = False
        if hasattr(websocket_client, "is_connected"):
            already_connected = bool(websocket_client.is_connected())

        if (
            auto_connect
            and not already_connected
            and hasattr(
                websocket_client,
                "connect",
            )
        ):
            websocket_client.connect(
                instrument_tokens=[token],
                mode=mode,
                threaded=True,
            )
        else:
            websocket_client.subscribe([token], mode=mode)

        self._stream_client = websocket_client
        self._stream_instrument_token = token
        self._stream_mode = mode

    def stop_stream(self) -> None:
        """Unsubscribe ticker token from active stream client, if available."""
        if self._stream_client is None or self._stream_instrument_token is None:
            return

        if hasattr(self._stream_client, "unsubscribe"):
            self._stream_client.unsubscribe([self._stream_instrument_token])

        self._stream_client = None
        self._stream_instrument_token = None

    def _resolve_stream_instrument_token(
        self,
        instrument_token: int | None,
    ) -> int:
        if instrument_token is not None:
            return int(instrument_token)

        if self.candle_api is not None:
            resolver = getattr(self.candle_api, "_get_instrument_token", None)
            if callable(resolver):
                resolved = resolver(self.name)
                if isinstance(resolved, (int, float, str)):
                    return int(resolved)

        raise ValueError(
            "instrument_token is required when candle_api cannot resolve it"
        )

    def _process_stream_ticks(
        self,
        ticks: Any,
        instrument_token: int,
    ) -> bool:
        """Apply realtime ticks to sequence and indicators.

        Returns:
            ``True`` when sequence was changed, otherwise ``False``.
        """
        if not isinstance(ticks, list) or not ticks:
            return False

        changed = False

        for tick in ticks:
            if not isinstance(tick, dict):
                continue

            tick_token = tick.get("instrument_token")
            if tick_token is None or int(tick_token) != int(instrument_token):
                continue

            price_raw = tick.get("last_price")
            if price_raw is None:
                continue

            price = float(price_raw)
            quantity = float(tick.get("last_traded_quantity") or 0.0)
            fallback_tzinfo = None
            if self.sequence.candles:
                last_ts = self.sequence.candles[-1].timestamp
                if isinstance(last_ts, datetime):
                    fallback_tzinfo = last_ts.tzinfo

            tick_time = self._extract_tick_timestamp(
                tick=tick,
                fallback_tzinfo=fallback_tzinfo,
            )
            candle_timestamp = self._bucket_timestamp(tick_time)

            changed = (
                self._apply_stream_point(
                    candle_timestamp=candle_timestamp,
                    price=price,
                    quantity=quantity,
                )
                or changed
            )

        return changed

    def _apply_stream_point(
        self,
        candle_timestamp: datetime,
        price: float,
        quantity: float,
    ) -> bool:
        if not self.sequence.candles:
            candle = self.candle_builder.build_candle(
                timestamp=candle_timestamp,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=quantity,
            )
            self.update_sequence(
                candle=candle,
                operation=UpdateOperation.APPEND,
            )
            self.sequence_capacity = len(self.sequence.candles)
            return True

        last = self.sequence.candles[-1]
        if not isinstance(last.timestamp, datetime):
            return False

        candle_timestamp, last_timestamp = self._align_timestamp_timezones(
            candle_timestamp,
            last.timestamp,
        )

        if candle_timestamp < last_timestamp:
            return False

        if candle_timestamp == last_timestamp:
            updated = self.candle_builder.build_candle(
                timestamp=last_timestamp,
                open=last.open,
                high=max(last.high, price),
                low=min(last.low, price),
                close=price,
                volume=last.volume + quantity,
            )
            self.update_sequence(
                candle=updated,
                operation=UpdateOperation.UPDATE,
            )
            return True

        new_candle = self.candle_builder.build_candle(
            timestamp=candle_timestamp,
            open=price,
            high=price,
            low=price,
            close=price,
            volume=quantity,
        )

        if (
            self.sequence_capacity is not None
            and self.sequence_capacity > 0
            and len(self.sequence.candles) >= self.sequence_capacity
        ):
            self.update_sequence(
                candle=new_candle,
                operation=UpdateOperation.ROLLING_ADD,
            )
        else:
            self.update_sequence(
                candle=new_candle,
                operation=UpdateOperation.APPEND,
            )

        return True

    def _extract_tick_timestamp(
        self,
        tick: Dict[str, Any],
        fallback_tzinfo: Any = None,
    ) -> datetime:
        for key in (
            "exchange_timestamp",
            "last_trade_time",
            "timestamp",
        ):
            value = tick.get(key)
            if isinstance(value, datetime):
                return value

        if fallback_tzinfo is not None:
            return datetime.now(tz=fallback_tzinfo)

        return datetime.now()

    def _align_timestamp_timezones(
        self,
        left: datetime,
        right: datetime,
    ) -> tuple[datetime, datetime]:
        """Align naive/aware datetime pairs for safe comparisons."""
        left_naive = left.tzinfo is None
        right_naive = right.tzinfo is None

        if left_naive and not right_naive:
            left = left.replace(tzinfo=right.tzinfo)
        elif right_naive and not left_naive:
            right = right.replace(tzinfo=left.tzinfo)

        return left, right

    def _bucket_timestamp(self, timestamp: datetime) -> datetime:
        interval_minutes = self._interval_minutes(self.interval)

        if interval_minutes >= 60 * 24:
            return timestamp.replace(
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )

        minutes = timestamp.hour * 60 + timestamp.minute
        bucket_start = (minutes // interval_minutes) * interval_minutes
        bucket_hour = bucket_start // 60
        bucket_minute = bucket_start % 60

        return timestamp.replace(
            hour=bucket_hour,
            minute=bucket_minute,
            second=0,
            microsecond=0,
        )

    def _interval_minutes(self, interval: str) -> int:
        normalized = interval.strip().lower()
        aliases = {
            "1m": 1,
            "minute": 1,
            "3m": 3,
            "3minute": 3,
            "5m": 5,
            "5minute": 5,
            "10m": 10,
            "10minute": 10,
            "15m": 15,
            "15minute": 15,
            "30m": 30,
            "30minute": 30,
            "60m": 60,
            "60minute": 60,
            "1h": 60,
            "day": 60 * 24,
            "1d": 60 * 24,
        }

        return aliases.get(normalized, 1)

    def _to_candle(self, candle: Candle | Dict[str, Any]) -> Candle:
        """Normalize input payload into a deep-copied ``Candle`` object."""
        if isinstance(candle, Candle):
            return copy.deepcopy(candle)

        return self.candle_builder.build_candle(
            candle["timestamp"],
            candle["open"],
            candle["high"],
            candle["low"],
            candle["close"],
            candle["volume"],
        )

    def build_sequence(
        self,
        candles: TypingSequence[Candle | Dict[str, Any]],
        recompute: bool = True,
    ) -> None:
        """Backward-compatible alias for sequence initialization."""
        self.initialize(candles=candles, recompute=recompute)

    def build_sequence_from_api(
        self,
        limit: int,
        recompute: bool = True,
    ) -> None:
        """Backward-compatible alias for API-based initialization."""
        self.initialize(candles=None, api_limit=limit, recompute=recompute)

    def add_indicator(
        self,
        indicator: Indicator,
        recompute: bool = True,
    ) -> None:
        """Register one indicator and optionally recompute all indicators."""
        if any(existing.name == indicator.name for existing in self.indicators):
            raise ValueError(f"Indicator '{indicator.name}' already exists")

        self.indicators.append(copy.deepcopy(indicator))
        if recompute:
            self.recompute_indicators()

    def add_indicators(
        self,
        indicators: List[Indicator],
        recompute: bool = True,
    ) -> None:
        """Register multiple indicators and optionally recompute once."""
        for indicator in indicators:
            self.add_indicator(indicator, recompute=False)

        if recompute:
            self.recompute_indicators()

    def remove_indicator(self, indicator_name: str) -> None:
        """Remove indicator registration and cached values by name."""
        self.indicators = [
            indicator
            for indicator in self.indicators
            if indicator.name != indicator_name
        ]
        self.indicator_values.pop(indicator_name, None)
        self._latest_indicator_values.pop(indicator_name, None)

    def get_indicator_value(self, name: str) -> IndicatorPoint | None:
        """Return latest computed point for one indicator, if available."""
        points = self.indicator_values.get(name)
        if not points:
            return None
        return points[-1]

    def get_indicator_points(self, name: str) -> List[IndicatorPoint]:
        """Return full computed point series for a single indicator."""
        return list(self.indicator_values.get(name, []))

    def get_all_indicator_values(self) -> Dict[str, List[IndicatorPoint]]:
        """Return deep-ish copies of all indicator point series."""
        return {name: list(points) for name, points in self.indicator_values.items()}

    def get_latest_indicator_values(self) -> Dict[str, float | None]:
        """Return latest scalar value per indicator for quick reads."""
        return dict(self._latest_indicator_values)

    def recompute_indicators(
        self,
        parallel: bool | None = None,
    ) -> Dict[str, List[IndicatorPoint]]:
        """Recompute all registered indicators for the current sequence state.

        Args:
            parallel: Force parallel/sequential mode when provided.
                If ``None``,
                auto-select mode using ``_should_parallel_compute``.

        Returns:
            Mapping of indicator name to full timestamped point series.
        """
        if not self.indicators:
            self.indicator_values = {}
            self._latest_indicator_values = {}
            return self.indicator_values

        if parallel is None:
            parallel = self._should_parallel_compute()

        if parallel:
            computed_points = self._recompute_indicators_parallel()
        else:
            computed_points = self._recompute_indicators_sequential()

        self.indicator_values = computed_points
        self._latest_indicator_values = {
            indicator.name: (
                computed_points[indicator.name][-1].value
                if computed_points[indicator.name]
                else None
            )
            for indicator in self.indicators
        }

        return self.get_all_indicator_values()

    def _should_parallel_compute(self) -> bool:
        """Return whether indicator workload merits thread-pool execution."""
        return (
            len(self.indicators) >= self.parallel_compute_threshold
            and len(self.sequence.candles) > 0
        )

    def _recompute_indicators_sequential(
        self,
    ) -> Dict[str, List[IndicatorPoint]]:
        """Compute indicator series one-by-one in the current thread."""
        return {
            indicator.name: indicator.compute(self.sequence)
            for indicator in self.indicators
        }

    def _recompute_indicators_parallel(
        self,
    ) -> Dict[str, List[IndicatorPoint]]:
        """Compute indicator series concurrently using a thread pool."""
        computed_values: Dict[str, List[IndicatorPoint]] = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    indicator.compute,
                    self.sequence,
                ): indicator.name
                for indicator in self.indicators
            }

            for future in as_completed(futures):
                indicator_name = futures[future]
                computed_values[indicator_name] = future.result()

        return computed_values

    def update_sequence(
        self,
        candle: Candle,
        operation: UpdateOperation | str = UpdateOperation.APPEND,
        position: int | None = None,
    ) -> None:
        """Apply one candle update to the underlying sequence.

        Supported operations:
        - append: append a new candle to the end
        - rolling_add: drop the oldest candle and add the new one
        - update: replace one existing candle, defaulting to the latest
        """
        resolved_operation = self._coerce_operation(operation)

        if resolved_operation == UpdateOperation.APPEND:
            self.sequence.append_candle(candle)
        elif resolved_operation == UpdateOperation.ROLLING_ADD:
            self.sequence.add_candle(candle)
        elif resolved_operation == UpdateOperation.UPDATE:
            if not self.sequence.candles:
                raise ValueError("Cannot update a candle in an empty sequence")

            target_position = (
                len(self.sequence.candles) - 1 if position is None else position
            )

            self.sequence.update_candle(target_position, candle)
        else:
            raise ValueError(
                "operation must be one of: 'append', 'rolling_add', 'update'"
            )

        self.recompute_indicators()

    def update_sequence_batch(
        self,
        candles: List[Candle],
        operation: UpdateOperation | str = UpdateOperation.APPEND,
    ) -> None:
        """Apply one append-style operation to multiple candles in sequence."""
        resolved_operation = self._coerce_operation(operation)

        if resolved_operation == UpdateOperation.UPDATE:
            raise ValueError(
                "Batch update does not support 'update'; use update_sequence "
                "with an explicit position"
            )

        for candle in candles:
            self.update_sequence(candle=candle, operation=resolved_operation)

    @staticmethod
    def _coerce_operation(operation: UpdateOperation | str) -> UpdateOperation:
        """Normalize string/enum inputs to ``UpdateOperation`` members."""
        if isinstance(operation, UpdateOperation):
            return operation

        try:
            return UpdateOperation(operation)
        except ValueError as exc:
            raise ValueError(
                "operation must be one of: 'append', 'rolling_add', 'update'"
            ) from exc

    def get_indicator(self, indicator_name: str) -> Indicator | None:
        """Retrieve a registered indicator by name."""
        for indicator in self.indicators:
            if indicator.name == indicator_name:
                return indicator
        return None

    def __repr__(self) -> str:
        """Return concise ticker representation for logging and debugging."""
        indicator_names = [indicator.name for indicator in self.indicators]
        return (
            f"Ticker(name='{self.name}', "
            f"sequence_length={len(self.sequence.candles)}, "
            f"indicators={indicator_names})"
        )
