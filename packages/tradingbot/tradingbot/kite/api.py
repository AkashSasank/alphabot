from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict

from tradingbot.core.candles import Candle
from tradingbot.core.constants import Interval
from tradingbot.core.protocols import CandleAPIProvider
from tradingbot.kite.session import KiteSession


class KiteCandleAPIProvider(CandleAPIProvider):
    """Candle provider implementation backed by Kite Connect."""

    def __init__(
        self,
        session: KiteSession,
        config: Dict[str, Any] | None = None,
    ):
        self.session = session
        self.config = config or {}
        self.default_exchange = str(self.config.get("default_exchange", "NSE"))
        self._instrument_token_cache: Dict[str, int] = {}
        self._interval_aliases: Dict[str, str] = {
            "1m": Interval.MINUTE,
            "3m": Interval.THREE_MINUTE,
            "5m": Interval.FIVE_MINUTE,
            "10m": Interval.TEN_MINUTE,
            "15m": Interval.FIFTEEN_MINUTE,
            "30m": Interval.THIRTY_MINUTE,
            "60m": Interval.SIXTY_MINUTE,
            "1h": Interval.SIXTY_MINUTE,
            "day": Interval.DAY,
            "1d": Interval.DAY,
        }

    def fetch_candles(
        self,
        symbol: str,
        interval: str,
        limit: int,
        as_of: datetime | None = None,
    ) -> list[Candle | Dict[str, Any]]:
        """Fetch candles from Kite and normalize shape for ticker usage."""
        if limit <= 0:
            return []

        kite_interval = self._normalize_interval(interval)
        instrument_token = self._get_instrument_token(symbol)

        end_time = as_of if as_of is not None else datetime.now()
        start_time = end_time - self._estimate_lookback(kite_interval, limit)
        payload = self.session.kite.historical_data(
            instrument_token=instrument_token,
            from_date=start_time,
            to_date=end_time,
            interval=kite_interval,
            continuous=False,
            oi=False,
        )

        if not payload:
            return []

        candles: list[Candle | Dict[str, Any]] = []
        for item in payload[-limit:]:
            candles.append(
                {
                    "timestamp": item["date"],
                    "open": float(item["open"]),
                    "high": float(item["high"]),
                    "low": float(item["low"]),
                    "close": float(item["close"]),
                    "volume": float(item["volume"]),
                }
            )
        return candles

    def _normalize_interval(self, interval: str) -> str:
        normalized = interval.strip().lower()
        if normalized in self._interval_aliases:
            return self._interval_aliases[normalized]
        return interval

    def _estimate_lookback(self, interval: str, limit: int) -> timedelta:
        minutes_per_candle = Interval.MINUTES_PER_CANDLE.get(interval, 1)
        return timedelta(minutes=minutes_per_candle * max(limit + 5, limit))

    def _get_instrument_token(self, symbol: str) -> int:
        cache_key = symbol.upper()
        cached = self._instrument_token_cache.get(cache_key)
        if cached is not None:
            return cached

        exchange, tradingsymbol = self._parse_symbol(symbol)
        instruments = self.session.kite.instruments(exchange=exchange)
        for instrument in instruments:
            instrument_symbol = str(instrument.get("tradingsymbol", "")).upper()
            if instrument_symbol == tradingsymbol:
                token = int(instrument["instrument_token"])
                self._instrument_token_cache[cache_key] = token
                return token

        message = (
            "Unable to resolve instrument token "
            f"for symbol='{symbol}' on exchange='{exchange}'"
        )
        raise ValueError(message)

    def _parse_symbol(self, symbol: str) -> tuple[str, str]:
        if ":" in symbol:
            exchange, tradingsymbol = symbol.split(":", maxsplit=1)
            return exchange.upper(), tradingsymbol.upper()
        return self.default_exchange.upper(), symbol.upper()
