"""Time helpers for Kite candle intervals."""

from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from tradingbot.core.constants import Interval

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN = time(9, 15)


def normalize_datetime(value: datetime) -> datetime:
    """Return a datetime normalized to Indian market time."""
    if value.tzinfo is None:
        return value.replace(tzinfo=IST)
    return value.astimezone(IST)


def timeframe_delta(timeframe: str) -> timedelta:
    """Return the duration represented by a candle timeframe."""
    normalized = Interval.normalize(timeframe)
    minutes = Interval.MINUTES_PER_CANDLE.get(normalized)
    if minutes is None:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return timedelta(minutes=minutes)


def candle_bucket(timestamp: datetime, timeframe: str) -> datetime:
    """Return the candle start timestamp for a reading."""
    timestamp = normalize_datetime(timestamp)
    normalized = Interval.normalize(timeframe)

    if normalized == Interval.DAY:
        return timestamp.replace(hour=9, minute=15, second=0, microsecond=0)

    delta = timeframe_delta(normalized)
    session_start = datetime.combine(timestamp.date(), MARKET_OPEN, tzinfo=IST)
    elapsed = timestamp - session_start
    bucket_number = int(elapsed.total_seconds() // delta.total_seconds())
    return session_start + bucket_number * delta


def is_same_candle(
    latest_candle_timestamp: datetime,
    reading_timestamp: datetime,
    timeframe: str,
) -> bool:
    """Return whether two datetimes belong to the same candle bucket."""
    return candle_bucket(latest_candle_timestamp, timeframe) == candle_bucket(
        reading_timestamp,
        timeframe,
    )
