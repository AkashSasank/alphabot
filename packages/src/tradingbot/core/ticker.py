"""Ticker orchestration for API-seeded and websocket-updated candle sequences."""

from __future__ import annotations

import csv
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


from tradingbot.core.constants import Interval
from tradingbot.kite import (
    KiteCandleAPIProvider,
    KiteWebSocketClient,
)
from tradingbot.kite.session import KiteSession
from tradingbot.core.sequence import sequence_builder
from tradingbot.core.candles import Candle, candle_builder
from tradingbot.core.indicators import BaseIndicator, CandleView, IndicatorCursor
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN = time(9, 15)
INITIAL_CANDLES = 60
HISTORIC_INCEPTION_START = datetime(1990, 1, 1, tzinfo=IST)


@dataclass(frozen=True)
class TickerData:
    """Snapshot payload for ticker candle and indicator data."""

    symbol: str
    timeframe: str
    candles: list[Candle]
    indicators: dict[str, Any] | None = None


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
        self.sequence = sequence_builder.build_sequence(
            candles=[],
            interval=self.timeframe,
        )
        self.indicators: dict[str, BaseIndicator] = {}
        self.indicator_cursors: dict[str, IndicatorCursor] = {}
        self._last_cumulative_volume_by_token = {}
        self.initilaize_sequence()
        self.websocket_client = KiteWebSocketClient(
            session=session,
            on_ticks=self.on_ticks,
            on_connect=self.on_connect,
            on_close=self.on_close,
            on_error=self.on_error,
            mode=KiteWebSocketClient.MODE_FULL,
        )
        self.websocket_client.connect(symbols=[self.symbol], threaded=True)

    def add_indicator(self, name: str, indicator: BaseIndicator) -> Ticker:
        self.indicators[name] = indicator
        self.indicator_cursors[name] = indicator.cursor(self.sequence)
        return self

    def initilaize_sequence(self):
        candles_data = self.candle_api_provider.fetch_candles(
            symbol=self.symbol,
            interval=self.timeframe,
            from_date=datetime.now(IST)
            - self.timeframe_delta(self.timeframe) * INITIAL_CANDLES,
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
        print(f"Connected websocket for {self.symbol}: {response}", flush=True)

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
        return self.candle_bucket(timestamp, interval)

    def get_historic_data(
        self,
        ticker_name: str | None = None,
        timeframe: str | None = None,
        start_date: date | datetime | str | None = None,
        end_date: date | datetime | str | None = None,
        csv_path: str | Path | None = None,
    ) -> Path:
        """Fetch historical candles and save the full candle payload as CSV.

        Args:
            ticker_name: Optional symbol override. Defaults to this ticker's symbol.
            timeframe: Optional interval override. Defaults to this ticker's timeframe.
            start_date: Optional inclusive start datetime/date. Defaults to inception.
            end_date: Optional inclusive end datetime/date. Defaults to now.
            csv_path: Optional explicit destination path.

        Returns:
            Path to the written CSV file.
        """
        symbol = ticker_name or self.symbol
        interval = timeframe or self.timeframe
        from_date = self.parse_historic_datetime(start_date)
        to_date = self.parse_historic_datetime(end_date, is_end=True)
        if from_date is None:
            from_date = HISTORIC_INCEPTION_START

        candles_data = self.candle_api_provider.fetch_candles(
            symbol=symbol,
            interval=interval,
            from_date=from_date,
            to_date=to_date,
        )
        candles = [candle_builder.build_candle(**data) for data in candles_data]

        destination = (
            Path(csv_path)
            if csv_path is not None
            else self.historic_csv_path(
                symbol=symbol,
                timeframe=interval,
                start_date=from_date,
                end_date=to_date,
            )
        )
        destination.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = self.historic_candle_fieldnames()
        with destination.open("w", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            for candle in candles:
                row = candle.to_dict()
                timestamp = row["timestamp"]
                if isinstance(timestamp, datetime):
                    row["timestamp"] = timestamp.isoformat()
                writer.writerow(row)

        return destination

    def update_historic_data_indicators(
        self,
        csv_path: str | Path,
        chunk_size: int = 1000,
    ) -> Path:
        """Calculate registered indicators and update a historical candle CSV.

        Indicator columns use the names passed to ``add_indicator``. Existing
        indicator columns with the same names are overwritten. Rows are computed
        and written in chunks to avoid keeping the entire CSV in memory.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        destination = Path(csv_path)
        indicator_fieldnames = list(self.indicators.keys())
        if not indicator_fieldnames:
            return destination

        historic_candles: list[CandleView] = []

        temp_path: Path | None = None
        try:
            with destination.open("r", newline="") as input_file:
                reader = csv.DictReader(input_file)
                if reader.fieldnames is None:
                    raise ValueError(f"Historical CSV has no header: {destination}")

                output_fieldnames = self.merge_csv_fieldnames(
                    list(reader.fieldnames),
                    indicator_fieldnames,
                )
                with tempfile.NamedTemporaryFile(
                    "w",
                    newline="",
                    delete=False,
                    dir=destination.parent,
                    prefix=f".{destination.stem}.",
                    suffix=".tmp",
                ) as output_file:
                    temp_path = Path(output_file.name)
                    writer = csv.DictWriter(output_file, fieldnames=output_fieldnames)
                    writer.writeheader()

                    row_buffer: list[dict[str, Any]] = []
                    for row in reader:
                        historic_candles.append(
                            CandleView(
                                timestamp=row["timestamp"],
                                open=float(row["open"]),
                                high=float(row["high"]),
                                low=float(row["low"]),
                                close=float(row["close"]),
                                volume=float(row["volume"]),
                            )
                        )
                        self.add_indicator_values_to_historic_row(row, historic_candles)
                        row_buffer.append(row)

                        if len(row_buffer) >= chunk_size:
                            writer.writerows(row_buffer)
                            row_buffer.clear()
                            output_file.flush()

                    if row_buffer:
                        writer.writerows(row_buffer)
                        output_file.flush()

            temp_path.replace(destination)
        except Exception:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()
            raise

        return destination

    def add_indicator_values_to_historic_row(
        self,
        row: dict[str, Any],
        candles: list[CandleView],
    ) -> None:
        """Compute latest registered indicator values into a historical row."""
        for indicator_name, indicator in self.indicators.items():
            point = indicator.compute_point(candles)
            row[indicator_name] = "" if point.value is None else point.value

    def read_historic_csv(
        self,
        csv_path: Path,
    ) -> tuple[list[dict[str, str]], list[str]]:
        """Read historical CSV rows and fieldnames."""
        with csv_path.open("r", newline="") as file_obj:
            reader = csv.DictReader(file_obj)
            return list(reader), list(reader.fieldnames or [])

    def build_candles_from_historic_rows(
        self,
        rows: list[dict[str, str]],
    ) -> list[Candle]:
        """Build candles from historical CSV rows."""
        candles: list[Candle] = []
        for row in rows:
            candles.append(self.build_candle_from_historic_row(row))
        return candles

    def build_candle_from_historic_row(self, row: dict[str, str]) -> Candle:
        """Build a candle from one historical CSV row."""
        timestamp = self.parse_historic_datetime(row["timestamp"])
        if timestamp is None:
            raise ValueError("Historical CSV row is missing a timestamp.")
        return candle_builder.build_candle(
            timestamp=timestamp,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )

    def write_historic_rows(
        self,
        csv_path: Path,
        rows: list[dict[str, Any]],
        fieldnames: list[str],
    ) -> None:
        """Write historical rows back to CSV."""
        with csv_path.open("w", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def merge_csv_fieldnames(
        self,
        existing_fieldnames: list[str],
        new_fieldnames: list[str],
    ) -> list[str]:
        """Append missing fieldnames while preserving existing CSV order."""
        merged = list(existing_fieldnames)
        for fieldname in new_fieldnames:
            if fieldname not in merged:
                merged.append(fieldname)
        return merged

    @staticmethod
    def historic_candle_fieldnames() -> list[str]:
        """Return base historical candle CSV fieldnames."""
        return [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "candle_type",
            "candle_color",
            "body_size",
            "upper_wick_size",
            "lower_wick_size",
            "candle_size",
            "upper_wick_to_candle_ratio",
            "lower_wick_to_candle_ratio",
            "upper_wick_to_body_ratio",
            "lower_wick_to_body_ratio",
            "wick_difference_to_candle_ratio",
            "body_to_candle_ratio",
        ]

    def parse_historic_datetime(
        self,
        value: date | datetime | str | None,
        *,
        is_end: bool = False,
    ) -> datetime | None:
        """Parse optional user-provided date values for historical queries."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, date):
            boundary = time.max if is_end else time.min
            return datetime.combine(value, boundary)
        if isinstance(value, str):
            raw_value = value.strip()
            if self.is_date_only(raw_value):
                parsed_date = date.fromisoformat(raw_value)
                boundary = time.max if is_end else time.min
                return datetime.combine(parsed_date, boundary)
            return datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
        raise TypeError(f"Unsupported date value: {value!r}")

    @staticmethod
    def is_date_only(value: str) -> bool:
        """Return whether a string is an ISO date without a time component."""
        return len(value) == 10 and value[4] == "-" and value[7] == "-"

    def historic_csv_path(
        self,
        *,
        symbol: str,
        timeframe: str,
        start_date: datetime | None,
        end_date: datetime | None,
    ) -> Path:
        """Build the default destination path for a historical data export."""
        normalized_interval = Interval.normalize(timeframe)
        start_label = self.datetime_filename_label(start_date) if start_date else "inception"
        end_label = self.datetime_filename_label(end_date) if end_date else "now"
        return (
            Path("historic_data")
            / f"{symbol.upper()}_{normalized_interval}_{start_label}_{end_label}.csv"
        )

    @staticmethod
    def datetime_filename_label(value: datetime) -> str:
        """Return a filesystem-safe datetime label."""
        return value.isoformat().replace(":", "-").replace("+", "plus")

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
