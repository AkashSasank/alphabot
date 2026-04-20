from __future__ import annotations

from typing import Any, Dict

from tradingbot.kite.session import KiteSession, KiteSessionManager


def _build_kite_ticker(
    api_key: str,
    access_token: str,
    debug: bool = False,
    reconnect: bool = True,
    reconnect_max_tries: int = 50,
    reconnect_max_delay: int = 60,
    connect_timeout: int = 30,
) -> Any:
    """Build and configure a KiteTicker instance from KiteConnect.

    Args:
        api_key: Kite API key
        access_token: Authenticated access token
        debug: Enable debug logging
        reconnect: Enable automatic reconnection
        reconnect_max_tries: Maximum reconnection attempts
        reconnect_max_delay: Max delay in seconds between retries
        connect_timeout: WebSocket connection timeout in seconds

    Returns:
        Configured KiteTicker instance
    """
    try:
        from kiteconnect import KiteTicker
    except ImportError:
        raise ImportError(
            "KiteConnect library not installed. "
            "Install with: pip install kiteconnect"
        )

    ticker = KiteTicker(
        api_key=api_key,
        access_token=access_token,
        debug=debug,
    )

    return ticker


class KiteWebSocketClient:
    """Wrapper around KiteTicker for realtime instrument updates."""

    MODE_LTP = "ltp"
    MODE_QUOTE = "quote"
    MODE_FULL = "full"

    def __init__(
        self,
        session: KiteSession,
        config: Dict[str, Any] | None = None,
    ) -> None:
        self.session = session
        self.config = config or {}
        self._startup_tokens: list[int] = []
        self._startup_mode = str(self.config.get("ws_mode", self.MODE_QUOTE))
        self._callbacks: Dict[str, Any] = {
            "ticks": None,
            "connect": None,
            "close": None,
            "error": None,
            "message": None,
            "reconnect": None,
            "noreconnect": None,
            "order_update": None,
        }

        self.ticker = _build_kite_ticker(
            api_key=self.session.api_key,
            access_token=self.session.access_token,
            debug=bool(self.config.get("ws_debug", False)),
            reconnect=bool(self.config.get("ws_reconnect", True)),
            reconnect_max_tries=int(self.config.get("ws_reconnect_max_tries", 50)),
            reconnect_max_delay=int(self.config.get("ws_reconnect_max_delay", 60)),
            connect_timeout=int(self.config.get("ws_connect_timeout", 30)),
        )
        self._wire_callbacks()

    def set_on_ticks(self, callback: Any) -> None:
        self._callbacks["ticks"] = callback

    def set_on_connect(self, callback: Any) -> None:
        self._callbacks["connect"] = callback

    def set_on_close(self, callback: Any) -> None:
        self._callbacks["close"] = callback

    def set_on_error(self, callback: Any) -> None:
        self._callbacks["error"] = callback

    def set_on_message(self, callback: Any) -> None:
        self._callbacks["message"] = callback

    def set_on_reconnect(self, callback: Any) -> None:
        self._callbacks["reconnect"] = callback

    def set_on_noreconnect(self, callback: Any) -> None:
        self._callbacks["noreconnect"] = callback

    def set_on_order_update(self, callback: Any) -> None:
        self._callbacks["order_update"] = callback

    def connect(
        self,
        instrument_tokens: list[int] | None = None,
        mode: str | None = None,
        threaded: bool | None = None,
        disable_ssl_verification: bool = False,
        proxy: Dict[str, Any] | None = None,
    ) -> None:
        """Connect websocket and optionally subscribe to tokens at startup."""
        if instrument_tokens:
            self._startup_tokens = list(
                dict.fromkeys(int(t) for t in instrument_tokens)
            )

        if mode:
            self._startup_mode = mode

        connect_threaded = (
            bool(self.config.get("ws_threaded", True)) if threaded is None else threaded
        )

        self.ticker.connect(
            threaded=connect_threaded,
            disable_ssl_verification=disable_ssl_verification,
            proxy=proxy,
        )

    def subscribe(
        self,
        instrument_tokens: list[int],
        mode: str | None = None,
    ) -> None:
        """Subscribe to tokens and optionally set stream mode."""
        if not instrument_tokens:
            return

        normalized = list(dict.fromkeys(int(t) for t in instrument_tokens))
        self.ticker.subscribe(normalized)
        resolved_mode = mode or self._startup_mode
        self.ticker.set_mode(resolved_mode, normalized)

    def unsubscribe(self, instrument_tokens: list[int]) -> None:
        if not instrument_tokens:
            return
        normalized = list(dict.fromkeys(int(t) for t in instrument_tokens))
        self.ticker.unsubscribe(normalized)

    def set_mode(self, mode: str, instrument_tokens: list[int]) -> None:
        if not instrument_tokens:
            return
        normalized = list(dict.fromkeys(int(t) for t in instrument_tokens))
        self.ticker.set_mode(mode, normalized)

    def close(
        self,
        code: int | None = None,
        reason: str | None = None,
    ) -> None:
        self.ticker.close(code=code, reason=reason)

    def stop(self) -> None:
        self.ticker.stop()

    def stop_retry(self) -> None:
        self.ticker.stop_retry()

    def is_connected(self) -> bool:
        return bool(self.ticker.is_connected())

    def _wire_callbacks(self) -> None:
        self.ticker.on_ticks = self._on_ticks
        self.ticker.on_connect = self._on_connect
        self.ticker.on_close = self._on_close
        self.ticker.on_error = self._on_error
        self.ticker.on_message = self._on_message
        self.ticker.on_reconnect = self._on_reconnect
        self.ticker.on_noreconnect = self._on_noreconnect
        self.ticker.on_order_update = self._on_order_update

    def _on_ticks(self, ws: Any, ticks: list[Dict[str, Any]]) -> None:
        callback = self._callbacks["ticks"]
        if callback:
            callback(self, ticks)

    def _on_connect(self, ws: Any, response: Any) -> None:
        if self._startup_tokens:
            self.subscribe(
                instrument_tokens=self._startup_tokens,
                mode=self._startup_mode,
            )

        callback = self._callbacks["connect"]
        if callback:
            callback(self, response)

    def _on_close(self, ws: Any, code: int, reason: str) -> None:
        callback = self._callbacks["close"]
        if callback:
            callback(self, code, reason)

    def _on_error(self, ws: Any, code: int, reason: str) -> None:
        callback = self._callbacks["error"]
        if callback:
            callback(self, code, reason)

    def _on_message(self, ws: Any, payload: Any, is_binary: bool) -> None:
        callback = self._callbacks["message"]
        if callback:
            callback(self, payload, is_binary)

    def _on_reconnect(self, ws: Any, attempts_count: int) -> None:
        callback = self._callbacks["reconnect"]
        if callback:
            callback(self, attempts_count)

    def _on_noreconnect(self, ws: Any) -> None:
        callback = self._callbacks["noreconnect"]
        if callback:
            callback(self)

    def _on_order_update(self, ws: Any, data: Dict[str, Any]) -> None:
        callback = self._callbacks["order_update"]
        if callback:
            callback(self, data)


def create_kite_session(config: Dict[str, Any]) -> KiteSession:
    """Create authenticated Kite session from config.

    Uses KiteSessionManager to authenticate and initialize session.

    Required config keys:
    - api_key
    - api_secret
    - user_id
    - password
    - pin

    Optional config keys:
    - headless (bool, default True)
    - timeout_ms (int, default 45000)
    - default_exchange (str, default NSE)
    - browser_args (list[str], passed to Chromium launch)
    """
    manager = KiteSessionManager(config)
    return manager.create_session()
