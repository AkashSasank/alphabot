"""Feature extraction utilities for historical ticker datasets.

This module provides a high-level extractor that:
- uses BotManager for authenticated Kite access
- downloads historical candles from inception to a target date
- computes all supported indicators on the full candle history
- flattens nested candle/indicator fields into CSV-ready rows
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List

from dotenv import load_dotenv
from tradingbot.bot import BotManager
from tradingbot.core.constants import Interval
from tradingbot.core.indicators import build_popular_indicators
from tradingbot.core.ticker import Ticker


def _sanitize_name(value: str) -> str:
    """Convert free-form names into stable snake_case-like fragments."""
    lowered = value.strip().lower()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    lowered = lowered.strip("_")
    return lowered or "field"


def _normalize_scalar(value: Any) -> Any:
    """Normalize values for CSV serialization."""
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _flatten_dict(payload: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested dictionaries into one level with underscores."""
    flattened: Dict[str, Any] = {}

    for key, value in payload.items():
        key_part = _sanitize_name(str(key))
        field_name = f"{prefix}_{key_part}" if prefix else key_part

        if isinstance(value, dict):
            flattened.update(_flatten_dict(value, prefix=field_name))
            continue

        if isinstance(value, list):
            # Preserve arrays as pipe-separated scalar text where possible.
            normalized_items = [_normalize_scalar(item) for item in value]
            flattened[field_name] = "|".join(str(item) for item in normalized_items)
            continue

        flattened[field_name] = _normalize_scalar(value)

    return flattened


class HistoricalFeatureExtractor:
    """Extract candle + indicator features and persist to CSV."""

    def __init__(self, manager: BotManager) -> None:
        self.manager = manager

    @staticmethod
    def build_all_indicators():
        """Return all currently supported indicator implementations."""
        return build_popular_indicators()

    def fetch_historical_candles(
        self,
        ticker_name: str,
        interval: str,
        start_date: datetime,
        end_date: datetime,
    ) -> List[Dict[str, Any]]:
        """Fetch full historical candles in chunks for API safety."""
        if start_date > end_date:
            raise ValueError("start_date must be less than or equal to end_date")

        kite_interval = self.manager.api._normalize_interval(interval)
        token = self.manager.api._get_instrument_token(ticker_name)
        chunk = self._chunk_timedelta(kite_interval)

        candles_by_timestamp: Dict[Any, Dict[str, Any]] = {}
        cursor = start_date

        while cursor <= end_date:
            window_end = min(cursor + chunk, end_date)

            payload, resolved_window_end = self._fetch_historical_chunk(
                token=token,
                from_date=cursor,
                to_date=window_end,
                interval=kite_interval,
            )

            for item in payload:
                ts = item["date"]
                candles_by_timestamp[ts] = {
                    "timestamp": ts,
                    "open": float(item["open"]),
                    "high": float(item["high"]),
                    "low": float(item["low"]),
                    "close": float(item["close"]),
                    "volume": float(item["volume"]),
                }

            cursor = resolved_window_end + timedelta(seconds=1)

        sorted_timestamps = sorted(candles_by_timestamp.keys())
        return [candles_by_timestamp[ts] for ts in sorted_timestamps]

    def build_feature_rows(
        self,
        ticker_name: str,
        interval: str,
        candles: Iterable[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Compute all indicators and flatten each row for CSV export."""
        indicators = self.build_all_indicators()

        ticker = Ticker(
            name=ticker_name,
            interval=interval,
            indicators=indicators,
            candle_api=self.manager.api,
        )
        ticker.initialize(candles=list(candles), recompute=True)

        indicator_values = ticker.get_all_indicator_values()
        rows: List[Dict[str, Any]] = []

        for index, candle in enumerate(ticker.sequence.candles):
            row: Dict[str, Any] = {
                "ticker": ticker_name,
                "interval": interval,
            }

            candle_flat = _flatten_dict(candle.model_dump())
            row.update(candle_flat)

            for indicator_name, points in indicator_values.items():
                safe_indicator_name = _sanitize_name(indicator_name)
                if index >= len(points):
                    row[f"indicator_{safe_indicator_name}_timestamp"] = None
                    row[f"indicator_{safe_indicator_name}_value"] = None
                    continue

                point = points[index]
                point_flat = _flatten_dict(
                    point.model_dump(),
                    prefix=f"indicator_{safe_indicator_name}",
                )
                row.update(point_flat)

            rows.append(row)

        return rows

    def export_features_csv(
        self,
        ticker_name: str,
        interval: str,
        output_csv_path: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Path:
        """Fetch history, build flattened features, and write CSV to disk."""
        candles = self.fetch_historical_candles(
            ticker_name=ticker_name,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
        )
        if not candles:
            raise ValueError("No historical candles returned for requested date range.")

        rows = self.build_feature_rows(
            ticker_name=ticker_name,
            interval=interval,
            candles=candles,
        )

        output_path = Path(output_csv_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        all_columns = sorted({key for row in rows for key in row.keys()})

        with output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=all_columns)
            writer.writeheader()
            writer.writerows(rows)

        return output_path

    @staticmethod
    def _chunk_timedelta(kite_interval: str) -> timedelta:
        """Return conservative chunk sizes for Kite historical downloads."""
        if kite_interval == Interval.DAY:
            # Keep this safely under the API max date-window constraint.
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
        token: int,
        from_date: datetime,
        to_date: datetime,
        interval: str,
    ) -> tuple[list[Dict[str, Any]], datetime]:
        """Fetch one chunk and auto-reduce on interval-limit errors."""
        current_to_date = to_date

        while True:
            try:
                payload = self.manager.session.kite.historical_data(
                    instrument_token=token,
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

    @staticmethod
    def _extract_max_limit_days(message: str) -> int | None:
        """Parse max-day window from Kite input exception text."""
        match = re.search(r"max limit:\s*(\d+)\s*days", message, re.IGNORECASE)
        if not match:
            return None
        return int(match.group(1))


def _parse_datetime(value: str | None, fallback: datetime) -> datetime:
    if not value:
        return fallback
    return datetime.fromisoformat(value)


def _build_session_config_from_env() -> Dict[str, Any]:
    return {
        "api_key": os.getenv("KITE_API_KEY"),
        "api_secret": os.getenv("KITE_API_SECRET"),
        "user_id": os.getenv("KITE_USER_ID"),
        "password": os.getenv("KITE_PASSWORD"),
        "pin": os.getenv("KITE_PIN"),
        "headless": os.getenv("KITE_HEADLESS", "False").lower() == "true",
        "timeout_ms": int(os.getenv("KITE_TIMEOUT_MS", "45000")),
        "redirect_url": os.getenv(
            "KITE_REDIRECT_URL",
            "http://localhost:1130/",
        ),
    }


def extract_features_command() -> None:
    """CLI entrypoint for flattened historical feature extraction."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract full historical candles with all indicators " "and export to CSV"
        )
    )
    parser.add_argument(
        "--ticker",
        required=True,
        help="Ticker symbol, e.g. SBIN",
    )
    parser.add_argument(
        "--interval",
        default="day",
        help="Candle interval (e.g. day, 60m, 15m, 5m)",
    )
    parser.add_argument(
        "--start-date",
        default="1996-01-01T00:00:00",
        help="Inclusive start datetime (ISO format)",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Inclusive end datetime (ISO format). Default: now",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Default: ./features_<ticker>_<interval>.csv",
    )

    args = parser.parse_args()

    load_dotenv()

    end_date = _parse_datetime(args.end_date, fallback=datetime.now())
    start_date = _parse_datetime(
        args.start_date,
        fallback=datetime(1996, 1, 1),
    )

    output = args.output
    if not output:
        safe_ticker = _sanitize_name(args.ticker)
        safe_interval = _sanitize_name(args.interval)
        output = f"features_{safe_ticker}_{safe_interval}.csv"

    manager = BotManager(_build_session_config_from_env())
    extractor = HistoricalFeatureExtractor(manager)
    output_path = extractor.export_features_csv(
        ticker_name=args.ticker,
        interval=args.interval,
        output_csv_path=output,
        start_date=start_date,
        end_date=end_date,
    )

    print(f"CSV export completed: {output_path}")


if __name__ == "__main__":
    extract_features_command()
