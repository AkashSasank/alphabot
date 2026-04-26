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

LOGGER = logging.getLogger(__name__)


class CandleCache(ABC):
    """Abstract storage interface for candle persistence."""

    @abstractmethod
    def save(self, candles: List[Dict[str, Any]]) -> None:
        """Save or merge candles into storage.

        Args:
            candles: List of candle dicts to persist.
        """
        pass

    @abstractmethod
    def load(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> List[Candle]:
        """Load candles from storage, optionally filtered by date range.

        Args:
            start_date: Inclusive start date (or Min if None).
            end_date: Inclusive end date (or Max if None).

        Returns:
            List of candle dicts sorted by timestamp.
        """
        pass

    @abstractmethod
    def get_earliest_timestamp(self) -> datetime | None:
        """Return earliest stored candle timestamp, or None if empty."""
        pass

    @abstractmethod
    def get_latest_timestamp(self) -> datetime | None:
        """Return latest stored candle timestamp, or None if empty."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all stored candles."""
        pass


class FileCandelStorage(CandleCache):
    """CSV-based file storage with in-memory index for fast queries."""

    def __init__(
        self,
        symbol: str,
        interval: str,
        storage_dir: str = ".candle_cache",
    ) -> None:
        """Initialize file storage with candle builder, symbol, and interval.

        Args:
            candle_builder: Builder instance for creating Candle objects.
            symbol: Ticker symbol (e.g., "SBIN").
            interval: Candle interval (e.g., "day", "5m").
            storage_dir: Root directory for storage files.
        """
        self.candle_builder = candle_builder
        self.symbol = symbol
        self.interval = interval
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.storage_dir / f"{symbol}_{interval}.csv"
        self.index_path = self.storage_dir / f"{symbol}_{interval}.index.json"

        # In-memory index for fast lookups
        self._candles: List[Candle] = []
        self._timestamp_index: Dict[int, int] = {}  # timestamp ms -> list index
        self._metadata: Dict[str, str] = {}  # Stores earliest and latest timestamps

        self._load_index_from_disk()

    def save(self, candles: List[Dict[str, Any]]) -> None:
        """Merge new candles into storage, avoiding duplicates."""
        if not candles:
            return

        candle_objects = [
            self.candle_builder.build_candle(
                c["timestamp"], c["open"], c["high"], c["low"], c["close"], c["volume"]
            )
            for c in candles
        ]

        # Merge candles by timestamp, replacing any existing candle for the same bucket.
        merged_by_ts = {c.timestamp: c for c in self._candles}
        for candle in candle_objects:
            merged_by_ts[candle.timestamp] = candle

        self._candles = sorted(merged_by_ts.values(), key=lambda c: c.timestamp)

        # Rebuild index
        self._rebuild_index()
        self._update_metadata()

        # Write to disk
        self._write_to_disk()
        self._write_index_to_disk()

        LOGGER.info(f"Saved {len(candles)} candles for {self.symbol}/{self.interval}")

    def load(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> List[Candle]:
        """Load candles from in-memory cache, filtered by date range."""
        if not self._candles:
            return []

        result = self._candles

        if start_date:
            result = [c for c in result if c.timestamp >= start_date]

        if end_date:
            result = [c for c in result if c.timestamp <= end_date]

        return result

    def get_earliest_timestamp(self) -> datetime | None:
        """Return earliest stored candle timestamp."""
        return self._candles[0].timestamp if self._candles else None

    def get_latest_timestamp(self) -> datetime | None:
        """Return latest stored candle timestamp."""
        return self._candles[-1].timestamp if self._candles else None

    def clear(self) -> None:
        """Clear all stored candles."""
        self._candles.clear()
        self._timestamp_index.clear()
        self._metadata.clear()
        if self.file_path.exists():
            self.file_path.unlink()
        if self.index_path.exists():
            self.index_path.unlink()
        LOGGER.info(f"Cleared storage for {self.symbol}/{self.interval}")

    def _rebuild_index(self) -> None:
        """Rebuild in-memory timestamp index."""
        self._timestamp_index.clear()
        for idx, candle in enumerate(self._candles):
            self._timestamp_index[int(candle.timestamp.timestamp() * 1000)] = idx

    def _update_metadata(self) -> None:
        """Update metadata with earliest and latest candle timestamps."""
        if self._candles:
            self._metadata["earliest_timestamp"] = self._candles[
                0
            ].timestamp.isoformat()
            self._metadata["latest_timestamp"] = self._candles[-1].timestamp.isoformat()

    def _load_index_from_disk(self) -> None:
        """Load index metadata and candles from disk if they exist."""
        # Load metadata first (faster)
        if self.index_path.exists():
            try:
                with open(self.index_path, "r") as f:
                    self._metadata = json.load(f)
                LOGGER.debug(
                    f"Loaded metadata index from {self.index_path}: {self._metadata}"
                )
            except Exception as e:
                LOGGER.warning(f"Failed to load index metadata: {e}")
                self._metadata = {}

        # Load full candle data
        self._load_candles_from_disk()

    def _load_candles_from_disk(self) -> None:
        """Load candles from CSV file if it exists."""
        if not self.file_path.exists():
            LOGGER.debug(f"Storage file not found: {self.file_path}")
            return

        try:
            with open(self.file_path, "r") as f:
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
                    self._candles.append(candle)

            self._rebuild_index()
            self._update_metadata()
            LOGGER.info(f"Loaded {len(self._candles)} candles from {self.file_path}")
        except Exception as e:
            LOGGER.error(f"Failed to load candles from {self.file_path}: {e}")
            self._candles.clear()

    def _write_index_to_disk(self) -> None:
        """Write metadata index to JSON file."""
        try:
            with open(self.index_path, "w") as f:
                json.dump(self._metadata, f)
            LOGGER.debug(f"Wrote metadata index to {self.index_path}")
        except Exception as e:
            LOGGER.error(f"Failed to write index metadata to {self.index_path}: {e}")

    def _write_to_disk(self) -> None:
        """Write all in-memory candles to CSV file."""
        try:
            with open(self.file_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["timestamp", "open", "high", "low", "close", "volume"],
                )
                writer.writeheader()
                for candle in self._candles:
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
            LOGGER.debug(f"Wrote {len(self._candles)} candles to {self.file_path}")
        except Exception as e:
            LOGGER.error(f"Failed to write candles to {self.file_path}: {e}")
