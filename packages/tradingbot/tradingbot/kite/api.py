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
        from_date: datetime | None = None,
        to_date: datetime | None = None,
    ) -> list[Dict[str, Any]]:
        """Fetch candles from Kite and normalize shape for ticker usage."""
        kite_interval = self._normalize_interval(interval)
        instrument_token = self._get_instrument_token(symbol)

        if from_date is None and to_date is None:
            end_time = self._now_like(None)
            start_time = end_time - self._estimate_lookback(kite_interval, 1)
        else:
            end_time = to_date or self._now_like(from_date)
            start_time = from_date or (
                end_time - self._estimate_lookback(kite_interval, 1)
            )

        start_time, end_time = self._align_datetime_awareness(start_time, end_time)

        if start_time > end_time:
            return []

        candle_dicts: Dict[datetime, Dict[str, Any]] = {}
        cursor = start_time
        chunk = self._chunk_timedelta(kite_interval)

        while cursor <= end_time:
            window_end = min(cursor + chunk, end_time)
            payload, resolved_window_end = self._fetch_historical_chunk(
                instrument_token=instrument_token,
                from_date=cursor,
                to_date=window_end,
                interval=kite_interval,
            )

            for item in payload:
                candle_dicts[item["date"]] = {
                    "timestamp": item["date"],
                    "open": float(item["open"]),
                    "high": float(item["high"]),
                    "low": float(item["low"]),
                    "close": float(item["close"]),
                    "volume": float(item["volume"]),
                }

            cursor = resolved_window_end + timedelta(seconds=1)

        sorted_timestamps = sorted(candle_dicts.keys())
        return [candle_dicts[timestamp] for timestamp in sorted_timestamps]

    @staticmethod
    def _now_like(reference: datetime | None) -> datetime:
        """Return current time matching reference awareness when possible."""
        if reference is not None and reference.tzinfo and reference.utcoffset() is not None:
            return datetime.now(reference.tzinfo)
        return datetime.now()

    @staticmethod
    def _align_datetime_awareness(
        start_time: datetime,
        end_time: datetime,
    ) -> tuple[datetime, datetime]:
        """Ensure both datetimes are either offset-aware or offset-naive."""
        start_is_aware = (
            start_time.tzinfo is not None and start_time.utcoffset() is not None
        )
        end_is_aware = end_time.tzinfo is not None and end_time.utcoffset() is not None

        if start_is_aware == end_is_aware:
            return start_time, end_time

        if start_is_aware:
            return start_time, end_time.replace(tzinfo=start_time.tzinfo)

        return start_time.replace(tzinfo=end_time.tzinfo), end_time

    def _normalize_interval(self, interval: str) -> str:
        normalized = interval.strip().lower()
        if normalized in self._interval_aliases:
            return self._interval_aliases[normalized]
        return interval

    def _estimate_lookback(self, interval: str, limit: int) -> timedelta:
        minutes_per_candle = Interval.MINUTES_PER_CANDLE.get(interval, 1)
        return timedelta(minutes=minutes_per_candle * max(limit + 5, limit))

    def _chunk_timedelta(self, kite_interval: str) -> timedelta:
        if kite_interval == Interval.DAY:
            return timedelta(days=1800)
        if kite_interval == Interval.SIXTY_MINUTE:
            return timedelta(days=365)
        if kite_interval in {
            Interval.MINUTE,
            Interval.THREE_MINUTE,
            Interval.FIVE_MINUTE,
            Interval.TEN_MINUTE,
            Interval.FIFTEEN_MINUTE,
            Interval.THIRTY_MINUTE,
        }:
            return timedelta(days=55)
        return timedelta(days=55)

    def _fetch_historical_chunk(
        self,
        instrument_token: int,
        from_date: datetime,
        to_date: datetime,
        interval: str,
    ) -> tuple[list[Dict[str, Any]], datetime]:
        current_to_date = to_date

        while True:
            try:
                payload = self.session.kite.historical_data(
                    instrument_token=instrument_token,
                    from_date=from_date,
                    to_date=current_to_date,
                    interval=interval,
                    continuous=False,
                    oi=False,
                )
                return payload, current_to_date
            except Exception as exc:
                message = str(exc)
                if "interval exceeds max limit" not in message.lower():
                    raise

                max_days = self._extract_max_limit_days(message)
                safe_days = max(
                    1,
                    (max_days - 1) if max_days is not None else 1999,
                )
                max_allowed_end = min(
                    from_date + timedelta(days=safe_days),
                    to_date,
                )

                if max_allowed_end <= from_date or max_allowed_end >= current_to_date:
                    raise

                current_to_date = max_allowed_end

    def _extract_max_limit_days(self, message: str) -> int | None:
        for part in message.split():
            try:
                value = int(part)
                if value > 0:
                    return value
            except ValueError:
                continue
        return None

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
