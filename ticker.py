import sys
import time
from pathlib import Path

from tradingbot.kite.time import candle_bucket


sys.path.insert(0, str(Path(__file__).resolve().parent / "packages" / "src"))

from tradingbot.core.constants import Interval
from tradingbot.kite import (
    KiteCandleAPIProvider,
    KiteSessionManager,
    KiteWebSocketClient,
)
from tradingbot.kite.session import KiteSession
from tradingbot.core.sequence import Sequence, sequence_builder
from tradingbot.core.candles import Candle, candle_builder
from tradingbot.core.indicators import BaseIndicator, IndicatorCursor
from tradingbot.core.ticker import TickerData

SYMBOL = "SBIN"
INTERVAL = Interval.MINUTE
INITIAL_CANDLES = 60


from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN = time(9, 15)


class Ticker:
    def __init__(
        self,
        symbol: str,
        timeframe: str,
        session: KiteSession,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.candle_api_provider = KiteCandleAPIProvider(session=session)
        self.websocket_client = KiteWebSocketClient(
            session=session,
            on_ticks=self.on_ticks,
            on_connect=self.on_connect,
            on_close=self.on_close,
            on_error=self.on_error,
            mode=KiteWebSocketClient.MODE_FULL,
        )
        self.websocket_client.connect(symbols=[self.symbol], threaded=True)
        self.sequence = sequence_builder.build_sequence(
            candles=[],
            interval=self.timeframe,
        )
        self.initilaize_sequence()
        self.indicators: dict[str, BaseIndicator] = {}
        self.indicator_cursors: dict[str, IndicatorCursor] = {}
        self._last_cumulative_volume_by_token = {}

    def add_indicator(self, name: str, indicator: BaseIndicator) -> Ticker:
        self.indicators[name] = indicator
        self.indicator_cursors[name] = indicator.cursor(self.sequence)
        return self
    
    def initilaize_sequence(self):
        candles_data = self.candle_api_provider.fetch_candles(
            symbol=self.symbol,
            interval=self.timeframe,
            from_date=datetime.now(IST)- self.timeframe_delta(self.timeframe) * INITIAL_CANDLES,
        )
        candles = [candle_builder.build_candle(**data) for data in candles_data]
        self.sequence.update_sequence(candles)

    def poll(self):
        if self.sequence.candles:
            if not self.is_same_candle(
                self.sequence.candles[-1].timestamp,
                self.now_like(self.sequence.candles[-1].timestamp),
                self.timeframe,
            ):
                candles_data = self.candle_api_provider.fetch_candles(
                    symbol=self.symbol,
                    interval=self.timeframe,
                    from_date=self.sequence.candles[-1].timestamp,
                )
                candles = [candle_builder.build_candle(**data) for data in candles_data]
                self.sequence.update_sequence(candles)
            return self.sequence.candles
        print("No candles in sequence, outside market hours")
        return []

    def on_ticks(self, client, ticks):
        for tick in ticks:
            candle = self.update_sequence_from_tick(
                self.sequence, tick, self._last_cumulative_volume_by_token
            )
            if candle is None:
                continue

    def on_connect(self, client, response):
        print(f"Connected websocket for {SYMBOL}: {response}", flush=True)

    def on_close(self, client, code, reason):
        print(f"Websocket closed: {code} - {reason}", flush=True)

    def on_error(self, client, code, reason):
        print(f"Websocket error: {code} - {reason}", flush=True)

    def update_sequence_from_tick(self, sequence, tick, last_volume_by_token):
        timestamp = self.tick_timestamp(tick)
        last_price = tick.get("last_price")
        if timestamp is None or last_price is None:
            return None

        candle_timestamp = self.bucket_timestamp(timestamp, sequence.interval)
        price = float(last_price)
        volume_delta = self.tick_volume_delta(tick, last_volume_by_token)

        if not sequence.candles:
            candle = candle_builder.build_candle(
                timestamp=candle_timestamp,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=volume_delta,
            )
            sequence.append_candle(candle)
            return candle

        latest = sequence.candles[-1]
        candle_timestamp = self.align_datetime_like(candle_timestamp, latest.timestamp)
        if candle_timestamp == latest.timestamp:
            candle = candle_builder.build_candle(
                timestamp=latest.timestamp,
                open=latest.open,
                high=max(latest.high, price),
                low=min(latest.low, price),
                close=price,
                volume=latest.volume + volume_delta,
            )
            sequence.update_candle(len(sequence.candles) - 1, candle)
            return candle

        return None

    def tick_volume_delta(self, tick, last_volume_by_token):
        token = tick.get("instrument_token")
        cumulative_volume = tick.get("volume_traded")
        if token is not None and cumulative_volume is not None:
            token_key = int(token)
            cumulative = float(cumulative_volume)
            previous = last_volume_by_token.get(token_key)
            last_volume_by_token[token_key] = cumulative
            if previous is not None and cumulative >= previous:
                return cumulative - previous

        quantity = (
            tick.get("last_traded_quantity")
            or tick.get("last_quantity")
            or tick.get("quantity")
            or 0
        )
        return float(quantity)

    def tick_timestamp(self, tick):
        value = (
            tick.get("timestamp")
            or tick.get("exchange_timestamp")
            or tick.get("last_trade_time")
        )
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            return datetime.fromisoformat(value)
        return None

    def bucket_timestamp(self, timestamp, interval):
        return candle_bucket(timestamp, interval)

    def align_datetime_like(self, value, reference):
        value_is_aware = value.tzinfo is not None and value.utcoffset() is not None
        reference_is_aware = (
            reference.tzinfo is not None and reference.utcoffset() is not None
        )

        if value_is_aware and reference_is_aware:
            return value.astimezone(reference.tzinfo)
        if value_is_aware and not reference_is_aware:
            return value.replace(tzinfo=None)
        if not value_is_aware and reference_is_aware:
            return value.replace(tzinfo=reference.tzinfo)
        return value

    def now_like(self, reference):
        if reference.tzinfo is not None and reference.utcoffset() is not None:
            return datetime.now(reference.tzinfo)
        return datetime.now()

    def interval_delta(self, interval):
        minutes = Interval.MINUTES_PER_CANDLE.get(Interval.normalize(interval), 1)
        return timedelta(minutes=minutes)

    def normalize_dt(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=IST)
        return value.astimezone(IST)

    def timeframe_delta(self, timeframe: str) -> timedelta:
        key = timeframe.strip().lower()

        values = {
            "1s": timedelta(seconds=1),
            "1second": timedelta(seconds=1),
            "1m": timedelta(minutes=1),
            "1minute": timedelta(minutes=1),
            "3m": timedelta(minutes=3),
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "30m": timedelta(minutes=30),
            "1h": timedelta(hours=1),
            "day": timedelta(days=1),
            "1d": timedelta(days=1),
            "week": timedelta(weeks=1),
            "1w": timedelta(weeks=1),
        }

        if key not in values:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        return values[key]

    def candle_bucket(self, timestamp: datetime, timeframe: str) -> datetime:
        timestamp = self.normalize_dt(timestamp)
        delta = self.timeframe_delta(timeframe)

        if timeframe.lower() in {"day", "1d"}:
            return timestamp.replace(hour=9, minute=15, second=0, microsecond=0)

        if timeframe.lower() in {"week", "1w"}:
            monday = timestamp.date() - timedelta(days=timestamp.weekday())
            return datetime.combine(monday, MARKET_OPEN, tzinfo=IST)

        session_start = datetime.combine(timestamp.date(), MARKET_OPEN, tzinfo=IST)
        elapsed = timestamp - session_start

        bucket_number = int(elapsed.total_seconds() // delta.total_seconds())
        return session_start + bucket_number * delta

    def is_same_candle(
        self,
        latest_candle_timestamp: datetime,
        reading_timestamp: datetime,
        timeframe: str,
    ) -> bool:
        return self.candle_bucket(
            latest_candle_timestamp, timeframe
        ) == self.candle_bucket(
            reading_timestamp,
            timeframe,
        )
