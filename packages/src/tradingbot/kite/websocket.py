from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import Any

from kiteconnect import KiteTicker

from tradingbot.kite.session import KiteSession

LOGGER = logging.getLogger(__name__)

TickCallback = Callable[["KiteWebSocketClient", list[dict[str, Any]]], None]
ConnectCallback = Callable[["KiteWebSocketClient", Any], None]
CloseCallback = Callable[["KiteWebSocketClient", int | None, str | None], None]
ErrorCallback = Callable[["KiteWebSocketClient", Any, Any], None]
OrderUpdateCallback = Callable[["KiteWebSocketClient", dict[str, Any]], None]


class KiteWebSocketClient:
    """WebSocket client backed by an authenticated Kite session."""

    MODE_LTP = KiteTicker.MODE_LTP
    MODE_QUOTE = KiteTicker.MODE_QUOTE
    MODE_FULL = KiteTicker.MODE_FULL

    def __init__(
        self,
        session: KiteSession,
        default_exchange: str = "NSE",
        mode: str = MODE_QUOTE,
        on_ticks: TickCallback | None = None,
        on_connect: ConnectCallback | None = None,
        on_close: CloseCallback | None = None,
        on_error: ErrorCallback | None = None,
        on_order_update: OrderUpdateCallback | None = None,
        debug: bool = False,
        reconnect: bool = True,
        reconnect_max_tries: int = KiteTicker.RECONNECT_MAX_TRIES,
        reconnect_max_delay: int = KiteTicker.RECONNECT_MAX_DELAY,
        connect_timeout: int = KiteTicker.CONNECT_TIMEOUT,
    ) -> None:
        self.session = session
        self.default_exchange = default_exchange
        self.mode = mode
        self.on_ticks = on_ticks
        self.on_connect = on_connect
        self.on_close = on_close
        self.on_error = on_error
        self.on_order_update = on_order_update
        self._tick_handlers: list[TickCallback] = []
        self._instrument_token_cache: dict[str, int] = {}
        self._pending_modes: dict[int, str] = {}

        api_key = self._session_value("api_key")
        access_token = self._session_value("access_token")

        self.ticker = KiteTicker(
            api_key=api_key,
            access_token=access_token,
            debug=debug,
            reconnect=reconnect,
            reconnect_max_tries=reconnect_max_tries,
            reconnect_max_delay=reconnect_max_delay,
            connect_timeout=connect_timeout,
        )
        self._bind_callbacks()

    @property
    def is_connected(self) -> bool:
        """Return whether the underlying KiteTicker is connected."""
        return bool(self.ticker.is_connected())

    def connect(
        self,
        instrument_tokens: Iterable[int] | None = None,
        symbols: Iterable[str] | None = None,
        mode: str | None = None,
        threaded: bool = True,
        disable_ssl_verification: bool = False,
        proxy: dict[str, Any] | None = None,
    ) -> None:
        """Open the websocket and subscribe to any provided tokens or symbols."""
        if instrument_tokens is not None:
            self.subscribe(instrument_tokens, mode=mode)

        if symbols is not None:
            self.subscribe_symbols(symbols, mode=mode)

        self.ticker.connect(
            threaded=threaded,
            disable_ssl_verification=disable_ssl_verification,
            proxy=proxy,
        )

    def subscribe(
        self,
        instrument_tokens: Iterable[int],
        mode: str | None = None,
    ) -> list[int]:
        """Subscribe to instrument tokens now, or after the socket connects."""
        tokens = self._normalize_tokens(instrument_tokens)
        if not tokens:
            return []

        selected_mode = mode or self.mode
        for token in tokens:
            self._pending_modes[token] = selected_mode

        if self.is_connected:
            self._apply_subscriptions(tokens=tokens, mode=selected_mode)

        return tokens

    def subscribe_symbols(
        self,
        symbols: Iterable[str],
        mode: str | None = None,
    ) -> list[int]:
        """Resolve trading symbols and subscribe to their instrument tokens."""
        tokens = [self.get_instrument_token(symbol) for symbol in symbols]
        return self.subscribe(tokens, mode=mode)

    def unsubscribe(self, instrument_tokens: Iterable[int]) -> list[int]:
        """Unsubscribe from instrument tokens and clear pending subscription state."""
        tokens = self._normalize_tokens(instrument_tokens)
        if not tokens:
            return []

        for token in tokens:
            self._pending_modes.pop(token, None)

        if self.is_connected:
            self.ticker.unsubscribe(tokens)

        return tokens

    def unsubscribe_symbols(self, symbols: Iterable[str]) -> list[int]:
        """Resolve trading symbols and unsubscribe from their instrument tokens."""
        tokens = [self.get_instrument_token(symbol) for symbol in symbols]
        return self.unsubscribe(tokens)

    def set_mode(
        self,
        mode: str,
        instrument_tokens: Iterable[int] | None = None,
    ) -> None:
        """Set stream mode for selected tokens, or all pending tokens."""
        source_tokens = (
            self._pending_modes.keys()
            if instrument_tokens is None
            else instrument_tokens
        )
        tokens = self._normalize_tokens(source_tokens)
        if not tokens:
            return

        for token in tokens:
            self._pending_modes[token] = mode

        if self.is_connected:
            self.ticker.set_mode(mode, tokens)

    def close(self, code: int | None = None, reason: str | None = None) -> None:
        """Close the websocket connection and stop reconnection retries."""
        self.ticker.close(code=code, reason=reason)

    def add_tick_handler(self, handler: TickCallback) -> None:
        """Register an additional callback to receive every tick batch."""
        self._tick_handlers.append(handler)

    def stop(self) -> None:
        """Stop KiteTicker's event loop."""
        self.ticker.stop()

    def get_instrument_token(self, symbol: str) -> int:
        """Resolve and cache Kite instrument token for a symbol."""
        exchange, tradingsymbol = self._parse_symbol(symbol)
        cache_key = f"{exchange}:{tradingsymbol}"
        cached = self._instrument_token_cache.get(cache_key)
        if cached is not None:
            return cached

        instruments = self.session.kite.instruments(exchange=exchange)
        for instrument in instruments:
            instrument_symbol = str(instrument.get("tradingsymbol", "")).upper()
            if instrument_symbol == tradingsymbol:
                token = int(instrument["instrument_token"])
                self._instrument_token_cache[cache_key] = token
                return token

        raise ValueError(
            "Unable to resolve instrument token "
            f"for symbol='{symbol}' on exchange='{exchange}'"
        )

    def _bind_callbacks(self) -> None:
        self.ticker.on_ticks = self._handle_ticks
        self.ticker.on_connect = self._handle_connect
        self.ticker.on_close = self._handle_close
        self.ticker.on_error = self._handle_error
        self.ticker.on_order_update = self._handle_order_update

    def _handle_connect(self, ticker: KiteTicker, response: Any) -> None:
        self._apply_pending_subscriptions()
        if self.on_connect is not None:
            self.on_connect(self, response)

    def _handle_ticks(self, ticker: KiteTicker, ticks: list[dict[str, Any]]) -> None:
        for handler in self._tick_handlers:
            handler(self, ticks)
        if self.on_ticks is not None:
            self.on_ticks(self, ticks)

    def _handle_close(
        self,
        ticker: KiteTicker,
        code: int | None,
        reason: str | None,
    ) -> None:
        if self.on_close is not None:
            self.on_close(self, code, reason)

    def _handle_error(self, ticker: KiteTicker, code: Any, reason: Any) -> None:
        if self.on_error is not None:
            self.on_error(self, code, reason)
        else:
            LOGGER.error("Kite websocket error: %s - %s", code, reason)

    def _handle_order_update(self, ticker: KiteTicker, data: dict[str, Any]) -> None:
        if self.on_order_update is not None:
            self.on_order_update(self, data)

    def _apply_pending_subscriptions(self) -> None:
        tokens_by_mode: dict[str, list[int]] = {}
        for token, mode in self._pending_modes.items():
            tokens_by_mode.setdefault(mode, []).append(token)

        for mode, tokens in tokens_by_mode.items():
            self._apply_subscriptions(tokens=tokens, mode=mode)

    def _apply_subscriptions(self, tokens: list[int], mode: str) -> None:
        self.ticker.subscribe(tokens)
        self.ticker.set_mode(mode, tokens)

    def _session_value(self, name: str) -> str:
        value = getattr(self.session.kite, name, None)
        if value is None:
            value = self.session.session_cache.get(name)
        if not value:
            raise ValueError(f"Kite session missing required value: {name}")
        return str(value)

    def _parse_symbol(self, symbol: str) -> tuple[str, str]:
        if ":" in symbol:
            exchange, tradingsymbol = symbol.split(":", maxsplit=1)
            return exchange.upper(), tradingsymbol.upper()
        return self.default_exchange.upper(), symbol.upper()

    @staticmethod
    def _normalize_tokens(instrument_tokens: Iterable[int]) -> list[int]:
        return [int(token) for token in instrument_tokens]
