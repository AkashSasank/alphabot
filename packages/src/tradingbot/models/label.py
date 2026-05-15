"""Dataset labelling utilities for raw candle data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd


BarrierTieBreak = Literal["lower", "upper"]
VerticalBarrierLabel = Literal["return", "neutral"]


class DatasetLabeller(Protocol):
    """Protocol for classes that label raw candle dataframes."""

    label_column: str

    def label(self, data: Any) -> Any:
        """Return a dataframe of labels aligned to ``data`` by index."""
        ...


@dataclass
class TripleBarrierLabeller:
    """Label candles using profit, stop-loss, and vertical time barriers.

    Labels are aligned with the input index:
    - ``1`` means the upper profit barrier was reached first, or the vertical
      barrier return is positive when ``vertical_barrier_label="return"``.
    - ``-1`` means the lower stop barrier was reached first, or the vertical
      barrier return is negative when ``vertical_barrier_label="return"``.
    - ``0`` means neutral/no movement, or no barrier hit when
      ``vertical_barrier_label="neutral"``.

    The final ``max_time_horizon`` rows cannot be labelled because they do not
    have enough future candles, so their label is ``NA``.
    """

    upper_profit_barrier: float
    lower_stop_barrier: float
    max_time_horizon: int
    price_column: str = "close"
    high_column: str = "high"
    low_column: str = "low"
    timestamp_column: str = "timestamp"
    label_column: str = "triple_barrier_label"
    return_column: str = "triple_barrier_return"
    horizon_column: str = "triple_barrier_horizon"
    event_column: str = "triple_barrier_event"
    tie_break: BarrierTieBreak = "lower"
    vertical_barrier_label: VerticalBarrierLabel = "return"
    include_metadata: bool = True

    def __post_init__(self) -> None:
        self._validate_configuration()

    def label(self, data: Any) -> Any:
        """Label a pandas or Dask dataframe of raw candles."""
        self._validate_input_columns(data)
        if _is_dask_dataframe(data):
            return self._label_dask(data)
        return self._label_pandas(data)

    @property
    def output_columns(self) -> list[str]:
        columns = [self.label_column]
        if self.include_metadata:
            columns.extend(
                [
                    self.return_column,
                    self.horizon_column,
                    self.event_column,
                ]
            )
        return columns

    def _label_dask(self, data: Any) -> Any:
        meta = self._empty_output_meta(data._meta)
        return data.map_overlap(
            self._label_pandas,
            before=0,
            after=self.max_time_horizon,
            meta=meta,
        )

    def _label_pandas(self, data: pd.DataFrame) -> pd.DataFrame:
        output = pd.DataFrame(index=data.index)
        if self.timestamp_column in data.columns:
            output[self.timestamp_column] = data[self.timestamp_column]

        row_count = len(data)
        close = data[self.price_column].astype(float).to_numpy()
        high = data[self.high_column].astype(float).to_numpy()
        low = data[self.low_column].astype(float).to_numpy()

        label = np.full(row_count, np.nan, dtype=float)
        label_return = np.full(row_count, np.nan, dtype=float)
        label_horizon = np.full(row_count, np.nan, dtype=float)
        event = np.full(row_count, None, dtype=object)

        if row_count:
            upper_barrier = close * (1.0 + (self.upper_profit_barrier / 100.0))
            lower_barrier = close * (1.0 - (self.lower_stop_barrier / 100.0))

            for step in range(1, self.max_time_horizon + 1):
                if step >= row_count:
                    break

                source = np.arange(0, row_count - step)
                undecided = np.isnan(label[source])
                upper_hit = high[step:] >= upper_barrier[:-step]
                lower_hit = low[step:] <= lower_barrier[:-step]
                hit = undecided & (upper_hit | lower_hit)
                if not hit.any():
                    continue

                hit_source = source[hit]
                both_hit = upper_hit[hit] & lower_hit[hit]
                upper_only = upper_hit[hit] & ~lower_hit[hit]

                step_label = np.where(upper_only, 1, -1)
                if both_hit.any() and self.tie_break == "upper":
                    step_label[both_hit] = 1

                label[hit_source] = step_label
                label_return[hit_source] = (
                    close[hit_source + step] / close[hit_source]
                ) - 1.0
                label_horizon[hit_source] = step
                event[hit_source] = np.where(step_label == 1, "upper", "lower")

            self._apply_vertical_barrier(close, label, label_return, label_horizon, event)

        output[self.label_column] = pd.Series(label, index=data.index).astype("Int64")

        if self.include_metadata:
            output[self.return_column] = label_return
            output[self.horizon_column] = pd.Series(
                label_horizon,
                index=data.index,
            ).astype("Int64")
            output[self.event_column] = event

        return output

    def _apply_vertical_barrier(
        self,
        close: np.ndarray,
        label: np.ndarray,
        label_return: np.ndarray,
        label_horizon: np.ndarray,
        event: np.ndarray,
    ) -> None:
        row_count = len(close)
        if row_count <= self.max_time_horizon:
            return

        source = np.arange(0, row_count - self.max_time_horizon)
        unresolved = np.isnan(label[source])
        if not unresolved.any():
            return

        unresolved_source = source[unresolved]
        exit_index = unresolved_source + self.max_time_horizon
        vertical_return = (close[exit_index] / close[unresolved_source]) - 1.0

        if self.vertical_barrier_label == "return":
            vertical_label = np.sign(vertical_return)
        else:
            vertical_label = np.zeros_like(vertical_return)

        label[unresolved_source] = vertical_label
        label_return[unresolved_source] = vertical_return
        label_horizon[unresolved_source] = self.max_time_horizon
        event[unresolved_source] = "vertical"

    def _empty_output_meta(self, input_meta: pd.DataFrame) -> pd.DataFrame:
        output = pd.DataFrame()
        if self.timestamp_column in input_meta.columns:
            output[self.timestamp_column] = pd.Series(
                dtype=input_meta[self.timestamp_column].dtype
            )
        output[self.label_column] = pd.Series(dtype="Int64")
        if self.include_metadata:
            output[self.return_column] = pd.Series(dtype="float64")
            output[self.horizon_column] = pd.Series(dtype="Int64")
            output[self.event_column] = pd.Series(dtype="object")
        return output

    def _validate_configuration(self) -> None:
        if not 0 <= self.upper_profit_barrier <= 100:
            raise ValueError("upper_profit_barrier must be a percentage in [0, 100].")
        if not 0 <= self.lower_stop_barrier <= 100:
            raise ValueError("lower_stop_barrier must be a percentage in [0, 100].")
        if self.max_time_horizon <= 0:
            raise ValueError("max_time_horizon must be greater than 0.")
        if self.tie_break not in {"lower", "upper"}:
            raise ValueError("tie_break must be either 'lower' or 'upper'.")
        if self.vertical_barrier_label not in {"return", "neutral"}:
            raise ValueError(
                "vertical_barrier_label must be either 'return' or 'neutral'."
            )

    def _validate_input_columns(self, data: Any) -> None:
        missing = [
            column
            for column in (self.price_column, self.high_column, self.low_column)
            if column not in data.columns
        ]
        if missing:
            raise ValueError(f"Cannot label candles; missing columns: {missing}.")


def build_labelled_feature_dataset(
    data: Any,
    feature_builder: Any,
    labeller: DatasetLabeller,
    *,
    drop_unlabelled: bool = True,
    feature_overlap: int | None = None,
) -> Any:
    """Combine feature extraction and labelling into one aligned dataset.

    ``feature_builder`` is expected to expose ``encode_dataframe`` like
    ``CandleFeatureBuilder``. The function accepts pandas or Dask dataframes and
    returns the matching dataframe type.
    """
    if _is_dask_dataframe(data):
        combined = _build_dask_labelled_feature_dataset(
            data,
            feature_builder,
            labeller,
            feature_overlap=feature_overlap,
        )
    else:
        features = feature_builder.encode_dataframe(data)
        labels = labeller.label(data)
        combined = _concat_frames(features, labels)

    if drop_unlabelled:
        combined = combined.dropna(subset=[labeller.label_column])
    return combined


def _build_dask_labelled_feature_dataset(
    data: Any,
    feature_builder: Any,
    labeller: DatasetLabeller,
    *,
    feature_overlap: int | None,
) -> Any:
    before = (
        _infer_feature_overlap(feature_builder)
        if feature_overlap is None
        else feature_overlap
    )
    after = getattr(labeller, "max_time_horizon", 0)
    meta = _build_labelled_feature_partition(
        data._meta,
        feature_builder=feature_builder,
        labeller=labeller,
    ).iloc[:0]
    return data.map_overlap(
        _build_labelled_feature_partition,
        before=before,
        after=after,
        meta=meta,
        feature_builder=feature_builder,
        labeller=labeller,
    )


def _build_labelled_feature_partition(
    data: pd.DataFrame,
    *,
    feature_builder: Any,
    labeller: DatasetLabeller,
) -> pd.DataFrame:
    features = feature_builder.encode_dataframe(data)
    labels = labeller.label(data)
    return _concat_frames(features, labels)


def _infer_feature_overlap(feature_builder: Any) -> int:
    output_columns = set(getattr(feature_builder, "output_columns", ()))
    candidates = [0]
    if "log_volume_zscore" in output_columns:
        candidates.append(getattr(feature_builder, "log_volume_window", 0))
    if "rsi" in output_columns:
        candidates.append(getattr(feature_builder, "rsi_period", 0))
    if "stochastic_rsi" in output_columns:
        candidates.append(
            getattr(feature_builder, "stochastic_rsi_period", 0)
            + getattr(feature_builder, "stochastic_rsi_stoch_period", 0)
        )
    if "macd" in output_columns:
        candidates.append(getattr(feature_builder, "macd_slow_period", 0))
    return max(candidates)


def _concat_frames(features: Any, labels: Any) -> Any:
    labels_to_join = labels.drop(
        columns=[column for column in labels.columns if column in features.columns],
    )
    if _is_dask_dataframe(features):
        import dask.dataframe as dd

        return dd.concat([features, labels_to_join], axis=1)
    return pd.concat([features, labels_to_join], axis=1)


def _is_dask_dataframe(data: Any) -> bool:
    return data.__class__.__module__.startswith("dask.dataframe")
