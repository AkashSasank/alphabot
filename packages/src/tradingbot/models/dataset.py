"""PyTorch datasets for Transformer sequence training."""

from __future__ import annotations

import bisect
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


TargetMode = Literal["class_index", "one_hot", "regression"]


class TransformerWindowDataset(Dataset):
    """Window labelled feature rows into ``(context, target)`` Transformer samples.

    This generalizes the ``CandleWindowDataset`` from ``notebooks/gpt.ipynb``:
    each item returns a feature window with shape ``context_length x feature_dim``
    and a target from a future row. With the default ``target_offset=1``, the
    target is the row immediately after the context window, matching the
    notebook's next-candle setup.
    """

    def __init__(
        self,
        csv_files: Iterable[str | Path],
        feature_columns: Sequence[str],
        context_length: int,
        *,
        label_columns: Sequence[str] | None = None,
        label_column: str | None = None,
        label_values: Sequence[int | float | str] | None = None,
        target_columns: Sequence[str] | None = None,
        target_mode: TargetMode = "class_index",
        target_offset: int = 1,
        max_rows_per_file: int | None = None,
        pickle_compression: str | None = "gzip",
    ):
        if context_length <= 0:
            raise ValueError("context_length must be greater than 0.")
        if target_offset < 0:
            raise ValueError("target_offset must be greater than or equal to 0.")
        if target_mode not in {"class_index", "one_hot", "regression"}:
            raise ValueError(
                "target_mode must be 'class_index', 'one_hot', or 'regression'."
            )
        if label_columns is not None and label_column is not None:
            raise ValueError("Use either label_columns or label_column, not both.")
        if target_columns is not None and (label_columns is not None or label_column is not None):
            raise ValueError("target_columns cannot be combined with label columns.")
        if target_mode in {"class_index", "one_hot"} and (
            label_columns is None and label_column is None
        ):
            raise ValueError("Classification datasets require label_columns or label_column.")
        if target_mode == "regression" and target_columns is None:
            raise ValueError("Regression datasets require target_columns.")

        self.csv_files = [Path(path) for path in csv_files]
        self.feature_columns = list(feature_columns)
        self.context_length = context_length
        self.label_columns = list(label_columns) if label_columns is not None else None
        self.label_column = label_column
        self.label_values = list(label_values) if label_values is not None else None
        self.target_columns = list(target_columns) if target_columns is not None else None
        self.target_mode = target_mode
        self.target_offset = target_offset
        self.max_rows_per_file = max_rows_per_file
        self.pickle_compression = pickle_compression

        self.feature_arrays: list[np.ndarray] = []
        self.target_arrays: list[np.ndarray] = []
        self.window_counts: list[int] = []
        self.cumulative_counts: list[int] = []
        self.skipped: list[tuple[str, list[str]]] = []
        self.label_counts: np.ndarray | None = None

        self._load_files()

    @property
    def target_dim(self) -> int:
        if self.target_mode == "regression":
            return len(self.target_columns or [])
        if self.label_columns is not None:
            return len(self.label_columns)
        return len(self.label_values or [])

    def __len__(self) -> int:
        return self.cumulative_counts[-1] if self.cumulative_counts else 0

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        file_index = bisect.bisect_right(self.cumulative_counts, index)
        previous_total = 0 if file_index == 0 else self.cumulative_counts[file_index - 1]
        start = index - previous_total
        target_index = start + self.context_length - 1 + self.target_offset

        x = self.feature_arrays[file_index][start : start + self.context_length].copy()
        y = self.target_arrays[file_index][target_index]

        if self.target_mode == "class_index":
            return torch.from_numpy(x), torch.tensor(int(y), dtype=torch.long)
        target = np.asarray(y, dtype=np.float32).copy()
        return torch.from_numpy(x), torch.from_numpy(target)

    def _load_files(self) -> None:
        running_total = 0
        class_labels = []

        for path in self.csv_files:
            frame = self._read_frame(path)
            required_columns = self._required_columns()
            missing = [column for column in required_columns if column not in frame.columns]
            if missing:
                self.skipped.append((path.name, missing))
                continue

            cleaned = (
                frame[required_columns]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            min_rows = self.context_length + self.target_offset
            if len(cleaned) < min_rows:
                self.skipped.append((path.name, ["not enough rows after cleaning"]))
                continue

            features = cleaned[self.feature_columns].to_numpy(dtype=np.float32)
            targets = self._extract_targets(cleaned)

            count = len(features) - self.context_length - self.target_offset + 1
            if count <= 0:
                self.skipped.append((path.name, ["not enough target rows"]))
                continue

            if self.target_mode == "class_index":
                usable_targets = targets[
                    self.context_length - 1 + self.target_offset :
                    self.context_length - 1 + self.target_offset + count
                ]
                class_labels.append(usable_targets.astype(np.int64))

            self.feature_arrays.append(features)
            self.target_arrays.append(targets)
            self.window_counts.append(count)
            running_total += count
            self.cumulative_counts.append(running_total)

        if self.target_mode == "class_index" and class_labels:
            labels = np.concatenate(class_labels)
            minlength = int(labels.max()) + 1 if labels.size else self.target_dim
            self.label_counts = np.bincount(labels, minlength=max(self.target_dim, minlength))

    def _read_frame(self, path: Path) -> pd.DataFrame:
        suffixes = [suffix.lower() for suffix in path.suffixes]
        if ".pkl" in suffixes or ".pickle" in suffixes:
            frame = pd.read_pickle(path, compression=self.pickle_compression)
            if self.max_rows_per_file is not None:
                return frame.head(self.max_rows_per_file)
            return frame
        return pd.read_csv(path, nrows=self.max_rows_per_file)

    def _required_columns(self) -> list[str]:
        columns = list(self.feature_columns)
        if self.target_mode == "regression":
            columns.extend(self.target_columns or [])
        elif self.label_columns is not None:
            columns.extend(self.label_columns)
        elif self.label_column is not None:
            columns.append(self.label_column)
        return list(dict.fromkeys(columns))

    def _extract_targets(self, frame: pd.DataFrame) -> np.ndarray:
        if self.target_mode == "regression":
            return frame[self.target_columns or []].to_numpy(dtype=np.float32)

        if self.label_columns is not None:
            labels = frame[self.label_columns].to_numpy(dtype=np.float32)
            if self.target_mode == "one_hot":
                return labels
            return labels.argmax(axis=1).astype(np.int64)

        scalar_labels = frame[self.label_column].to_numpy()
        if self.label_values is None:
            values = sorted(pd.unique(scalar_labels))
        else:
            values = self.label_values
        value_to_index = {value: index for index, value in enumerate(values)}
        class_index = np.array([value_to_index[value] for value in scalar_labels], dtype=np.int64)

        if self.target_mode == "class_index":
            return class_index

        one_hot = np.zeros((len(class_index), len(values)), dtype=np.float32)
        one_hot[np.arange(len(class_index)), class_index] = 1.0
        return one_hot


class CandleWindowDataset(TransformerWindowDataset):
    """Backward-compatible dataset wrapper from the GPT notebook."""

    def __init__(
        self,
        csv_files: Iterable[str | Path],
        feature_columns: Sequence[str],
        context_length: int,
        color_indices: Sequence[int],
        max_rows_per_file: int | None = None,
        pickle_compression: str | None = "gzip",
    ):
        color_feature_columns = [feature_columns[index] for index in color_indices]
        super().__init__(
            csv_files=csv_files,
            feature_columns=feature_columns,
            context_length=context_length,
            label_columns=color_feature_columns,
            target_mode="class_index",
            target_offset=1,
            max_rows_per_file=max_rows_per_file,
            pickle_compression=pickle_compression,
        )
        self.color_indices = list(color_indices)
        self.arrays = self.feature_arrays
