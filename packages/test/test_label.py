import pandas as pd
import pandas.testing as pdt
import pytest

from tradingbot.core.features import CandleFeatureBuilder
from tradingbot.models import (
    TripleBarrierLabeller,
    build_labelled_feature_dataset,
)


def _raw_candles():
    return pd.DataFrame(
        [
            {
                "timestamp": "2026-05-15T09:15:00+05:30",
                "open": 100,
                "high": 101,
                "low": 99,
                "close": 100,
                "volume": 1000,
            },
            {
                "timestamp": "2026-05-15T09:16:00+05:30",
                "open": 100,
                "high": 106,
                "low": 99,
                "close": 100,
                "volume": 1100,
            },
            {
                "timestamp": "2026-05-15T09:17:00+05:30",
                "open": 100,
                "high": 101,
                "low": 94,
                "close": 100,
                "volume": 1200,
            },
            {
                "timestamp": "2026-05-15T09:18:00+05:30",
                "open": 100,
                "high": 104,
                "low": 99,
                "close": 103,
                "volume": 1300,
            },
            {
                "timestamp": "2026-05-15T09:19:00+05:30",
                "open": 103,
                "high": 104,
                "low": 102,
                "close": 104,
                "volume": 1400,
            },
            {
                "timestamp": "2026-05-15T09:20:00+05:30",
                "open": 104,
                "high": 104,
                "low": 103,
                "close": 104,
                "volume": 1500,
            },
        ]
    )


def test_triple_barrier_labeller_labels_raw_candles():
    labeller = TripleBarrierLabeller(
        upper_profit_barrier=5,
        lower_stop_barrier=5,
        max_time_horizon=2,
    )

    labels = labeller.label(_raw_candles())

    assert labels["timestamp"].tolist() == _raw_candles()["timestamp"].tolist()
    assert labels["triple_barrier_label_lower"].tolist() == [0, 1, 0, 0, pd.NA, pd.NA]
    assert labels["triple_barrier_label_neutral"].tolist() == [0, 0, 0, 0, pd.NA, pd.NA]
    assert labels["triple_barrier_label_upper"].tolist() == [1, 0, 1, 1, pd.NA, pd.NA]


def test_triple_barrier_labeller_can_use_neutral_vertical_barrier():
    labeller = TripleBarrierLabeller(
        upper_profit_barrier=5,
        lower_stop_barrier=5,
        max_time_horizon=2,
        vertical_barrier_label="neutral",
    )

    labels = labeller.label(_raw_candles())

    assert labels["triple_barrier_label_lower"].tolist() == [0, 1, 0, 0, pd.NA, pd.NA]
    assert labels["triple_barrier_label_neutral"].tolist() == [0, 0, 1, 1, pd.NA, pd.NA]
    assert labels["triple_barrier_label_upper"].tolist() == [1, 0, 0, 0, pd.NA, pd.NA]


def test_triple_barrier_labeller_can_output_fixed_horizon_regression_target():
    labeller = TripleBarrierLabeller(
        upper_profit_barrier=5,
        lower_stop_barrier=5,
        max_time_horizon=2,
        output_type="regression",
    )

    labels = labeller.label(_raw_candles())

    assert labels.columns.tolist() == ["timestamp", "triple_barrier_return"]
    assert labels["triple_barrier_return"].iloc[:4].tolist() == pytest.approx(
        [0.0, 0.03, 0.04, (104 / 103) - 1.0]
    )
    assert labels["triple_barrier_return"].iloc[4:].isna().all()


def test_build_labelled_feature_dataset_combines_features_and_labels():
    feature_builder = CandleFeatureBuilder(
        [
            "timestamp",
            "upper_wick_to_candle_ratio",
            "lower_wick_to_candle_ratio",
            "body_to_candle_ratio",
        ]
    )
    labeller = TripleBarrierLabeller(
        upper_profit_barrier=5,
        lower_stop_barrier=5,
        max_time_horizon=2,
    )

    dataset = build_labelled_feature_dataset(_raw_candles(), feature_builder, labeller)

    assert dataset.columns.tolist() == [
        "timestamp",
        "upper_wick_to_candle_ratio",
        "lower_wick_to_candle_ratio",
        "body_to_candle_ratio",
        "triple_barrier_label_lower",
        "triple_barrier_label_neutral",
        "triple_barrier_label_upper",
    ]
    assert len(dataset) == 4
    assert dataset[
        [
            "triple_barrier_label_lower",
            "triple_barrier_label_neutral",
            "triple_barrier_label_upper",
        ]
    ].to_numpy().tolist() == [[0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 0, 1]]


def test_triple_barrier_labeller_handles_dask_partition_overlap():
    dd = pytest.importorskip("dask.dataframe")
    raw = _raw_candles()
    dask_raw = dd.from_pandas(raw, npartitions=3)
    labeller = TripleBarrierLabeller(
        upper_profit_barrier=5,
        lower_stop_barrier=5,
        max_time_horizon=2,
    )

    pandas_labels = labeller.label(raw)
    dask_labels = labeller.label(dask_raw).compute()

    pdt.assert_frame_equal(dask_labels, pandas_labels, check_dtype=False)


def test_build_labelled_feature_dataset_handles_dask_dataframe():
    dd = pytest.importorskip("dask.dataframe")
    feature_builder = CandleFeatureBuilder(
        [
            "timestamp",
            "upper_wick_to_candle_ratio",
            "lower_wick_to_candle_ratio",
            "body_to_candle_ratio",
        ]
    )
    labeller = TripleBarrierLabeller(
        upper_profit_barrier=5,
        lower_stop_barrier=5,
        max_time_horizon=2,
    )

    pandas_dataset = build_labelled_feature_dataset(
        _raw_candles(),
        feature_builder,
        labeller,
    )
    dask_dataset = build_labelled_feature_dataset(
        dd.from_pandas(_raw_candles(), npartitions=3),
        feature_builder,
        labeller,
    ).compute()

    pdt.assert_frame_equal(dask_dataset, pandas_dataset, check_dtype=False)
