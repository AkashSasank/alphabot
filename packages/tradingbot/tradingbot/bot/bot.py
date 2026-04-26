"""Bot manager orchestrating multiple trading bots with shared components."""

from __future__ import annotations

from datetime import datetime, timedelta
from time import sleep
from typing import Any, Callable, Dict, List

from pydantic import BaseModel
from tradingbot.core.candles import Candle, CandleBuilder
from tradingbot.core.indicators import Indicator
from tradingbot.core.sequence import IndicatorPoint, Sequence
from tradingbot.core.ticker import Ticker
from tradingbot.kite.api import KiteCandleAPIProvider
from tradingbot.kite.session import KiteSession, KiteSessionManager
from tradingbot.kite.socket import KiteWebSocketClient


class BotPollData(BaseModel):
    """Pydantic poll response returned by bot and manager poll interfaces."""

    interval: str
    candles: List[Candle]
    last_candle: Candle | None
    indicators: Dict[str, List[IndicatorPoint]]


class Bot:
    """Individual trading bot instance with ticker and callbacks.

    Each bot manages a single ticker and executes callbacks when
    sequence or indicators are updated.
    """

    def __init__(
        self,
        ticker: Ticker,
        on_sequence_update: Callable[[Sequence], None] | None = None,
        on_indicators_update: (
            Callable[[Dict[str, List[IndicatorPoint]]], None] | None
        ) = None,
    ) -> None:
        """Initialize a bot with ticker and optional callbacks.

        Args:
            ticker: Ticker instance managing instrument candles and indicators.
            on_sequence_update: Optional callback for sequence updates.
                Receives the updated Sequence object.
            on_indicators_update: Optional callback for indicators recompute.
                Receives a dict mapping indicator name to computed points.
        """
        self.ticker = ticker
        self.on_sequence_update = on_sequence_update
        self.on_indicators_update = on_indicators_update

    def emit_sequence_update(self) -> None:
        """Emit sequence update to registered callback."""
        if self.on_sequence_update:
            self.on_sequence_update(self.ticker.sequence)

    def emit_indicators_update(self) -> None:
        """Emit indicators update to registered callback."""
        if self.on_indicators_update:
            self.on_indicators_update(self.ticker.indicator_values)

    def poll(
        self,
        limit: int | None = None,
        emit_updates: bool = True,
        date: datetime | None = None,
    ) -> BotPollData:
        """Poll via ticker and optionally emit callbacks.

        Args:
            limit: Optional poll fetch size override for this ticker.
                When omitted, ticker-owned inherent limit is used.
            emit_updates: Emit callbacks after poll completes.
            date: Optional poll time passed to ticker polling.

        Returns:
            Pydantic payload containing candles, interval, and indicators.
        """
        before_length = len(self.ticker.sequence.candles)
        before_candle = (
            self.ticker.sequence.candles[-1] if self.ticker.sequence.candles else None
        )
        before_indicators = self.ticker.get_latest_indicator_values()

        if limit is not None:
            self.ticker.poll_limit = int(limit)

        self.ticker.poll(date=date)

        if emit_updates:
            after_length = len(self.ticker.sequence.candles)
            after_candle = (
                self.ticker.sequence.candles[-1]
                if self.ticker.sequence.candles
                else None
            )
            after_indicators = self.ticker.get_latest_indicator_values()

            candle_changed = (
                before_length != after_length or before_candle != after_candle
            )
            indicators_changed = before_indicators != after_indicators

            if candle_changed or indicators_changed:
                self.emit_sequence_update()
                self.emit_indicators_update()

        return self.build_poll_data()

    def build_poll_data(self) -> BotPollData:
        """Build a stable pydantic response payload for bot state consumers."""
        candles = [
            candle.model_copy(deep=True) for candle in self.ticker.sequence.candles
        ]
        last_candle = candles[-1] if candles else None
        indicators = {
            name: [point.model_copy(deep=True) for point in points]
            for name, points in self.ticker.get_all_indicator_values().items()
        }

        return BotPollData(
            interval=self.ticker.sequence.interval,
            candles=candles,
            last_candle=last_candle,
            indicators=indicators,
        )

    def stream(
        self,
        websocket_client: Any,
        instrument_token: int | None = None,
        mode: str = "quote",
        auto_connect: bool = True,
    ) -> None:
        """Start low-latency websocket streaming via ticker stream API."""
        self.ticker.stream(
            websocket_client=websocket_client,
            instrument_token=instrument_token,
            mode=mode,
            auto_connect=auto_connect,
        )

    def stop_stream(self) -> None:
        """Stop websocket streaming for this bot ticker."""
        self.ticker.stop_stream()


class BotManager:
    """Manages multiple trading bots with shared session, API, and socket.

    The manager coordinates:
    - Single authenticated Kite session
    - Shared candle API provider
    - Shared WebSocket client for real-time updates
    - Multiple bot instances, each with a ticker and callbacks

    Example:
        manager = BotManager(session_config)
        manager.add_bot_and_subscribe("infy_5m", indicators=[...])
        # manager.socket connects and streams real-time ticks
    """

    def __init__(
        self,
        api_config: Dict[str, Any] | None = None,
        socket_config: Dict[str, Any] | None = None,
    ) -> None:
        """Initialize BotManager with shared components.

        Session configuration is composed internally by KiteSessionManager
        from environment variables or user prompts.

        Args:
            api_config: Optional configuration for KiteCandleAPIProvider.
                Optional keys: default_exchange
            socket_config: Optional configuration for KiteWebSocketClient.
                Optional keys: ws_mode, ws_debug, ws_reconnect, etc.
        """
        self.api_config = api_config or {}
        self.socket_config = socket_config or {}

        # Initialize session with internal config composition
        session_manager = KiteSessionManager()

        # Check for cached token first
        if (
            hasattr(session_manager, "has_valid_cached_token")
            and session_manager.has_valid_cached_token()
        ):
            print("✅ Using cached access token - no login needed!")
            self.session: KiteSession = session_manager.create_session()
        else:
            # Run full login flow if no valid cached token
            print("\n" + "=" * 60)
            print("🔐 RUNNING FULL LOGIN FLOW")
            print("=" * 60)
            self.session: KiteSession = session_manager.create_session()

        # Initialize shared API and socket
        self.api: KiteCandleAPIProvider = KiteCandleAPIProvider(
            self.session, self.api_config
        )
        self.socket: KiteWebSocketClient = KiteWebSocketClient(
            self.session, self.socket_config
        )

        # Track bots by name
        self.bots: Dict[str, Bot] = {}
        self._backtest_next_poll_at: Dict[str, datetime] = {}
        self._backtest_started: Dict[str, bool] = {}
        self._backtest_limit: Dict[str, int] = {}

        # Wire socket ticks callback to update bots
        self.socket.set_on_ticks(self._on_socket_ticks)

    def add_bot_and_subscribe(
        self,
        ticker_name: str,
        interval: str = "5m",
        indicators: List[Indicator] | None = None,
        api_limit: int = 100,
        on_sequence_update: Callable[[Sequence], None] | None = None,
        on_indicators_update: (
            Callable[[Dict[str, List[IndicatorPoint]]], None] | None
        ) = None,
    ) -> Bot:
        """Add bot, initialize ticker, and subscribe to real-time updates.

        Args:
            ticker_name: Symbol/instrument identifier (e.g., "INFY", "TCS").
            interval: Candle interval (e.g., "5m", "15m", "1h").
            indicators: Optional list of indicator objects to register.
            api_limit: Number of historical candles to fetch at initialization.
            on_sequence_update: Optional callback for sequence updates.
            on_indicators_update: Optional callback for indicator recomputes.

        Returns:
            Bot: The newly created and subscribed bot instance.
        """
        # Create ticker with shared API and indicators
        ticker = Ticker(
            name=ticker_name,
            interval=interval,
            indicators=indicators,
            candle_api=self.api,
        )

        # Initialize ticker with historical candles
        ticker.initialize(api_limit=api_limit, recompute=True)

        # Create bot instance
        bot = Bot(
            ticker=ticker,
            on_sequence_update=on_sequence_update,
            on_indicators_update=on_indicators_update,
        )

        # Register bot
        bot_id = f"{ticker_name}_{interval}"
        self.bots[bot_id] = bot
        self._backtest_next_poll_at.pop(bot_id, None)
        self._backtest_started.pop(bot_id, None)
        self._backtest_limit.pop(bot_id, None)

        return bot

    def _on_socket_ticks(
        self,
        ws_or_ticks: Any,
        ticks: List[Dict[str, Any]] | None = None,
    ) -> None:
        """Handle real-time tick updates from WebSocket.

        Supports both callback styles:
        - callback(ticks)
        - callback(ws_client, ticks)

        Invoked when new ticks arrive. Updates all registered bots with
        latest candle data and recomputes indicators.

        Args:
            ws_or_ticks: WebSocket client or ticks payload.
            ticks: Optional ticks payload when ws_or_ticks is client.
        """
        resolved_ticks = ws_or_ticks if ticks is None else ticks
        if not isinstance(resolved_ticks, list):
            return

        for tick in resolved_ticks:
            instrument_token = tick.get("instrument_token")
            timestamp = tick.get("timestamp")
            ltp = tick.get("last_price")

            if ltp is None:
                continue

            # Update bots subscribed to this token
            for bot in self.bots.values():
                ticker = bot.ticker
                token = self.api._get_instrument_token(ticker.name)
                if token == instrument_token:
                    # Update candle with latest tick data
                    if (
                        ticker.sequence.candles
                        and ticker.sequence.candles[-1].timestamp == timestamp
                    ):
                        # Create updated candle with new close price
                        last_candle = ticker.sequence.get_candle(-1)
                        builder = CandleBuilder()
                        updated = builder.build_candle(
                            timestamp=last_candle.timestamp,
                            open=last_candle.open,
                            high=last_candle.high,
                            low=last_candle.low,
                            close=float(ltp),
                            volume=last_candle.volume,
                        )
                        ticker.sequence.update_candle(-1, updated)

                        # Recompute indicators and emit updates
                        ticker.recompute_indicators()
                        bot.emit_sequence_update()
                        bot.emit_indicators_update()

    def connect_socket(self) -> None:
        """Connect the WebSocket for real-time data streaming."""
        instrument_tokens = []
        for bot in self.bots.values():
            token = self.api._get_instrument_token(bot.ticker.name)
            instrument_tokens.append(token)

        if instrument_tokens:
            self.socket.connect(
                instrument_tokens=list(set(instrument_tokens)), threaded=True
            )

    def disconnect_socket(self) -> None:
        """Disconnect the WebSocket."""
        if self.socket.ticker:
            self.socket.ticker.close()

    def get_bot(self, bot_id: str) -> Bot | None:
        """Retrieve a bot by identifier.

        Args:
            bot_id: Bot identifier (format: "SYMBOL_INTERVAL").

        Returns:
            Bot instance or None if not found.
        """
        return self.bots.get(bot_id)

    def list_bots(self) -> List[str]:
        """Return list of all bot identifiers."""
        return list(self.bots.keys())

    def poll_bot(
        self,
        bot_id: str,
        limit: int | None = None,
        emit_updates: bool = True,
        date: datetime | None = None,
        backtest: bool = False,
        backtest_delay_seconds: float = 2.0,
    ) -> BotPollData:
        """Poll a specific bot ticker through the manager.

        Args:
            bot_id: Bot identifier (e.g. ``SBIN_5m``).
            limit: Optional poll fetch size override for this ticker.
                When omitted, ticker-owned inherent limit is used.
            emit_updates: Whether callbacks should be emitted.
            date: Optional poll time passed to ticker polling.
            backtest: Enables sequential date-based polling mode.
                In backtest mode, first call should include ``date``.
                Subsequent calls can omit ``date`` and manager advances
                the poll date by ticker timeframe each step.
            backtest_delay_seconds: Delay applied between backtest steps
                after the initial seed poll.

        Returns:
            Pydantic payload containing sequence, last candle, and indicators.
        """
        bot = self.get_bot(bot_id)
        if bot is None:
            raise ValueError(f"Bot '{bot_id}' not found")

        if not backtest:
            return bot.poll(limit=limit, emit_updates=emit_updates, date=date)

        if date is not None:
            self._seed_backtest_state(
                bot_id=bot_id,
                bot=bot,
                start_date=date,
                limit=limit,
            )
            self._backtest_started[bot_id] = True
            if emit_updates:
                bot.emit_sequence_update()
                bot.emit_indicators_update()
            return bot.build_poll_data()

        if bot_id not in self._backtest_next_poll_at:
            raise ValueError(
                "Backtest mode requires an initial date. "
                "Call poll_bot(..., backtest=True, date=<datetime>) first."
            )

        if self._backtest_started.get(bot_id, False) and backtest_delay_seconds > 0:
            sleep(backtest_delay_seconds)

        poll_date = self._backtest_next_poll_at[bot_id]
        single_candle_payload = self._fetch_candles_for_date(
            bot=bot,
            limit=1,
            date=poll_date,
        )

        if single_candle_payload:
            bot.ticker.apply_polled_candles(single_candle_payload)
            if emit_updates:
                bot.emit_sequence_update()
                bot.emit_indicators_update()

        payload = bot.build_poll_data()

        step = self._interval_to_timedelta(bot.ticker.interval)
        self._backtest_next_poll_at[bot_id] = poll_date + step
        self._backtest_started[bot_id] = True

        return payload

    def poll_all_bots(
        self,
        limit: int | None = None,
        emit_updates: bool = True,
        date: datetime | None = None,
        backtest: bool = False,
        backtest_delay_seconds: float = 2.0,
    ) -> Dict[str, BotPollData]:
        """Poll all registered bots through the manager.

        Args:
            limit: Optional poll fetch size override for each ticker.
                When omitted, ticker-owned inherent limit is used.
            emit_updates: Whether callbacks should be emitted.
            date: Optional poll time passed to ticker polling.
            backtest: Enables sequential date-based polling mode per bot.
            backtest_delay_seconds: Delay applied between backtest steps
                after the initial seed poll.

        Returns:
            Mapping from bot id to pydantic poll payload.
        """
        payload: Dict[str, BotPollData] = {}
        for index, bot in enumerate(self.bots.values()):
            bot_id = f"{bot.ticker.name}_{bot.ticker.interval}"
            delay_for_bot = backtest_delay_seconds if index == 0 else 0.0
            payload[bot_id] = self.poll_bot(
                bot_id=bot_id,
                limit=limit,
                emit_updates=emit_updates,
                date=date,
                backtest=backtest,
                backtest_delay_seconds=delay_for_bot,
            )

        return payload

    @staticmethod
    def _interval_to_timedelta(interval: str) -> timedelta:
        """Convert interval aliases to a timedelta step for backtesting."""
        normalized = interval.strip().lower()
        minute_aliases = {
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

        minutes = minute_aliases.get(normalized, 1)
        return timedelta(minutes=minutes)

    def _seed_backtest_state(
        self,
        bot_id: str,
        bot: Bot,
        start_date: datetime,
        limit: int | None,
    ) -> None:
        """Seed ticker state with an initial history window for backtesting."""
        resolved_limit = self._resolve_backtest_limit(bot, limit)
        step = self._interval_to_timedelta(bot.ticker.interval)
        initial_payload = self._fetch_candles_for_date(
            bot=bot,
            limit=resolved_limit,
            date=start_date,
        )

        bot.ticker.initialize(candles=initial_payload, recompute=True)
        bot.ticker.poll_limit = resolved_limit
        bot.ticker.sequence_capacity = max(
            resolved_limit,
            len(bot.ticker.sequence.candles),
        )

        self._backtest_limit[bot_id] = resolved_limit
        self._backtest_next_poll_at[bot_id] = start_date + step

    def _resolve_backtest_limit(
        self,
        bot: Bot,
        limit: int | None,
    ) -> int:
        """Resolve stable history window size for backtest replay."""
        if limit is not None and limit > 0:
            return int(limit)

        if bot.ticker.poll_limit is not None and bot.ticker.poll_limit > 0:
            return int(bot.ticker.poll_limit)

        if (
            bot.ticker.sequence_capacity is not None
            and bot.ticker.sequence_capacity > 0
        ):
            return int(bot.ticker.sequence_capacity)

        sequence_length = len(bot.ticker.sequence.candles)
        if sequence_length > 0:
            return sequence_length

        return 100

    def _fetch_candles_for_date(
        self,
        bot: Bot,
        limit: int,
        date: datetime,
    ) -> List[Candle | Dict[str, Any]]:
        """Fetch historical candles for a specific backtest timestamp."""
        interval_delta = self._interval_to_timedelta(bot.ticker.interval)
        from_date = date - interval_delta * max(limit + 5, limit)
        return self.api.fetch_candles(
            symbol=bot.ticker.name,
            interval=bot.ticker.interval,
            from_date=from_date,
            to_date=date,
        )
