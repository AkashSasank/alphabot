"""Storage abstraction for candle persistence and retrieval.

Provides pluggable storage backends for saving/loading historical candles
with indexing support for efficient range queries.
"""

import csv
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from tradingbot.core.candles import Candle, candle_builder
from tradingbot.core.constants import Interval

LOGGER = logging.getLogger(__name__)


class CandleCache(ABC):
    """Abstract storage interface for candle persistence."""

    @abstractmethod
    def save(
        self,
        candles: List[Dict[str, Any]],
        *,
        interval: str,
    ) -> None:
        """Save or merge candles into storage.

        Args:
            candles: List of candle dicts to persist.
            interval: Candle interval for the supplied candles.
        """
        pass

    @abstractmethod
    def load(
        self,
        *,
        interval: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> List[Candle]:
        """Load candles from storage, optionally filtered by date range.

        Args:
            interval: Candle interval to load.
            start_date: Inclusive start date (or Min if None).
            end_date: Inclusive end date (or Max if None).

        Returns:
            List of candle dicts sorted by timestamp.
        """
        pass

    @abstractmethod
    def get_earliest_timestamp(self, *, interval: str) -> datetime | None:
        """Return earliest stored candle timestamp, or None if empty."""
        pass

    @abstractmethod
    def get_latest_timestamp(self, *, interval: str) -> datetime | None:
        """Return latest stored candle timestamp, or None if empty."""
        pass

    @abstractmethod
    def clear(self, interval: str | None = None) -> None:
        """Clear all stored candles."""
        pass


class FileCandelStorage(CandleCache):
    """CSV-based file storage with in-memory index for fast queries."""

    def __init__(
        self,
        symbol: str,
        storage_dir: str = ".candle_cache",
    ) -> None:
        """Initialize file storage with candle builder and symbol.

        Args:
            symbol: Ticker symbol (e.g., "SBIN").
            storage_dir: Root directory for storage files.
        """
        self.candle_builder = candle_builder
        self.symbol = symbol
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self._candles_by_interval: Dict[str, List[Candle]] = {}
        self._timestamp_indexes: Dict[str, Dict[int, int]] = {}
        self._metadata_by_interval: Dict[str, Dict[str, str]] = {}
        self._loaded_intervals: set[str] = set()

    def save(
        self,
        candles: List[Dict[str, Any]],
        *,
        interval: str,
    ) -> None:
        """Merge new candles into storage, avoiding duplicates."""
        interval = self._resolve_interval(interval)
        if not candles:
            return

        self._ensure_loaded(interval)

        candle_objects = [
            self.candle_builder.build_candle(
                c["timestamp"], c["open"], c["high"], c["low"], c["close"], c["volume"]
            )
            for c in candles
        ]

        # Merge candles by timestamp, replacing any existing candle for the same bucket.
        current_candles = self._candles_by_interval.setdefault(interval, [])
        merged_by_ts = {c.timestamp: c for c in current_candles}
        for candle in candle_objects:
            merged_by_ts[candle.timestamp] = candle

        self._candles_by_interval[interval] = sorted(
            merged_by_ts.values(),
            key=lambda c: c.timestamp,
        )

        # Rebuild index
        self._rebuild_index(interval)
        self._update_metadata(interval)

        # Write to disk
        self._write_to_disk(interval)
        self._write_index_to_disk(interval)

        LOGGER.info(f"Saved {len(candles)} candles for {self.symbol}/{interval}")

    def load(
        self,
        *,
        interval: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> List[Candle]:
        """Load candles from in-memory cache, filtered by date range."""
        interval = self._resolve_interval(interval)
        self._ensure_loaded(interval)

        candles = self._candles_by_interval.get(interval, [])
        if not candles:
            return []

        result = candles

        if start_date:
            result = [c for c in result if c.timestamp >= start_date]

        if end_date:
            result = [c for c in result if c.timestamp <= end_date]

        return result

    def get_earliest_timestamp(self, *, interval: str) -> datetime | None:
        """Return earliest stored candle timestamp."""
        interval = self._resolve_interval(interval)
        self._ensure_loaded(interval)
        candles = self._candles_by_interval.get(interval, [])
        return candles[0].timestamp if candles else None

    def get_latest_timestamp(self, *, interval: str) -> datetime | None:
        """Return latest stored candle timestamp."""
        interval = self._resolve_interval(interval)
        self._ensure_loaded(interval)
        candles = self._candles_by_interval.get(interval, [])
        return candles[-1].timestamp if candles else None

    def clear(self, interval: str | None = None) -> None:
        """Clear all stored candles."""
        if interval is None:
            intervals = set(self._loaded_intervals)
            intervals.update(self._disk_intervals())
        else:
            intervals = {self._resolve_interval(interval)}

        for resolved_interval in intervals:
            self._candles_by_interval.pop(resolved_interval, None)
            self._timestamp_indexes.pop(resolved_interval, None)
            self._metadata_by_interval.pop(resolved_interval, None)
            self._loaded_intervals.discard(resolved_interval)
            file_path = self._file_path(resolved_interval)
            index_path = self._index_path(resolved_interval)
            if file_path.exists():
                file_path.unlink()
            if index_path.exists():
                index_path.unlink()
            LOGGER.info(f"Cleared storage for {self.symbol}/{resolved_interval}")

    def _resolve_interval(self, interval: str | None) -> str:
        if interval is not None:
            return self._normalize_interval(interval)
        raise ValueError("interval is required for symbol-level candle storage.")

    @staticmethod
    def _normalize_interval(interval: str) -> str:
        normalized = Interval.normalize(interval)
        if normalized not in Interval.MINUTES_PER_CANDLE:
            raise ValueError(f"Unsupported interval: {interval}")
        return normalized

    def _disk_intervals(self) -> set[str]:
        intervals: set[str] = set()
        prefix = f"{self.symbol}_"

        for path in self.storage_dir.iterdir():
            name = path.name
            if not name.startswith(prefix):
                continue
            if name.endswith(".index.json"):
                interval = name[len(prefix) : -len(".index.json")]
            elif name.endswith(".csv"):
                interval = name[len(prefix) : -len(".csv")]
            else:
                continue
            if interval in Interval.MINUTES_PER_CANDLE:
                intervals.add(interval)

        return intervals

    def _file_path(self, interval: str) -> Path:
        return self.storage_dir / f"{self.symbol}_{interval}.csv"

    def _index_path(self, interval: str) -> Path:
        return self.storage_dir / f"{self.symbol}_{interval}.index.json"

    def _ensure_loaded(self, interval: str) -> None:
        if interval not in self._loaded_intervals:
            self._load_interval_from_disk(interval)

    def _rebuild_index(self, interval: str) -> None:
        """Rebuild in-memory timestamp index."""
        timestamp_index = self._timestamp_indexes.setdefault(interval, {})
        timestamp_index.clear()
        for idx, candle in enumerate(self._candles_by_interval.get(interval, [])):
            timestamp_index[int(candle.timestamp.timestamp() * 1000)] = idx

    def _update_metadata(self, interval: str) -> None:
        """Update metadata with earliest and latest candle timestamps."""
        candles = self._candles_by_interval.get(interval, [])
        metadata = self._metadata_by_interval.setdefault(interval, {})
        if candles:
            metadata["earliest_timestamp"] = candles[0].timestamp.isoformat()
            metadata["latest_timestamp"] = candles[-1].timestamp.isoformat()
        else:
            metadata.clear()

    def _load_interval_from_disk(self, interval: str) -> None:
        """Load index metadata and candles from disk if they exist."""
        self._candles_by_interval.setdefault(interval, [])
        self._timestamp_indexes.setdefault(interval, {})
        self._metadata_by_interval.setdefault(interval, {})
        self._load_index_from_disk(interval)
        self._load_candles_from_disk(interval)
        self._loaded_intervals.add(interval)

    def _load_index_from_disk(self, interval: str) -> None:
        index_path = self._index_path(interval)
        # Load metadata first (faster)
        if index_path.exists():
            try:
                with open(index_path, "r") as f:
                    self._metadata_by_interval[interval] = json.load(f)
                metadata = self._metadata_by_interval[interval]
                LOGGER.debug(f"Loaded metadata index from {index_path}: {metadata}")
            except Exception as e:
                LOGGER.warning(f"Failed to load index metadata: {e}")
                self._metadata_by_interval[interval] = {}

    def _load_candles_from_disk(self, interval: str) -> None:
        """Load candles from CSV file if it exists."""
        file_path = self._file_path(interval)
        if not file_path.exists():
            LOGGER.debug(f"Storage file not found: {file_path}")
            return

        try:
            candles: List[Candle] = []
            with open(file_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    timestamp = datetime.fromisoformat(row["timestamp"])
                    candle = self.candle_builder.build_candle(
                        timestamp=timestamp,
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row["volume"]),
                    )
                    candles.append(candle)

            self._candles_by_interval[interval] = candles
            self._rebuild_index(interval)
            self._update_metadata(interval)
            LOGGER.info(f"Loaded {len(candles)} candles from {file_path}")
        except Exception as e:
            LOGGER.error(f"Failed to load candles from {file_path}: {e}")
            self._candles_by_interval[interval] = []

    def _write_index_to_disk(self, interval: str) -> None:
        """Write metadata index to JSON file."""
        index_path = self._index_path(interval)
        try:
            with open(index_path, "w") as f:
                json.dump(self._metadata_by_interval.get(interval, {}), f)
            LOGGER.debug(f"Wrote metadata index to {index_path}")
        except Exception as e:
            LOGGER.error(f"Failed to write index metadata to {index_path}: {e}")

    def _write_to_disk(self, interval: str) -> None:
        """Write all in-memory candles to CSV file."""
        file_path = self._file_path(interval)
        try:
            with open(file_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["timestamp", "open", "high", "low", "close", "volume"],
                )
                writer.writeheader()
                for candle in self._candles_by_interval.get(interval, []):
                    writer.writerow(
                        {
                            "timestamp": candle.timestamp.isoformat(),
                            "open": candle.open,
                            "high": candle.high,
                            "low": candle.low,
                            "close": candle.close,
                            "volume": candle.volume,
                        }
                    )
            count = len(self._candles_by_interval.get(interval, []))
            LOGGER.debug(f"Wrote {count} candles to {file_path}")
        except Exception as e:
            LOGGER.error(f"Failed to write candles to {file_path}: {e}")
