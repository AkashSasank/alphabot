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
    assert labels["triple_barrier_label"].tolist() == [1, -1, 1, 1, pd.NA, pd.NA]
    assert labels["triple_barrier_horizon"].tolist() == [1, 1, 2, 2, pd.NA, pd.NA]
    assert labels["triple_barrier_event"].iloc[:4].tolist() == [
        "upper",
        "lower",
        "vertical",
        "vertical",
    ]
    assert labels["triple_barrier_event"].iloc[4:].isna().all()


def test_triple_barrier_labeller_can_use_neutral_vertical_barrier():
    labeller = TripleBarrierLabeller(
        upper_profit_barrier=5,
        lower_stop_barrier=5,
        max_time_horizon=2,
        vertical_barrier_label="neutral",
    )

    labels = labeller.label(_raw_candles())

    assert labels["triple_barrier_label"].tolist() == [1, -1, 0, 0, pd.NA, pd.NA]


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
        "triple_barrier_label",
        "triple_barrier_return",
        "triple_barrier_horizon",
        "triple_barrier_event",
    ]
    assert len(dataset) == 4
    assert dataset["triple_barrier_label"].tolist() == [1, -1, 1, 1]


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

    pdt.assert_frame_equal(
        dask_labels.drop(columns=["triple_barrier_event"]),
        pandas_labels.drop(columns=["triple_barrier_event"]),
        check_dtype=False,
    )
    assert dask_labels["triple_barrier_event"].iloc[:4].tolist() == pandas_labels[
        "triple_barrier_event"
    ].iloc[:4].tolist()
    assert dask_labels["triple_barrier_event"].iloc[4:].isna().all()


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

    pdt.assert_frame_equal(
        dask_dataset.drop(columns=["triple_barrier_event"]),
        pandas_dataset.drop(columns=["triple_barrier_event"]),
        check_dtype=False,
    )
    assert dask_dataset["triple_barrier_event"].tolist() == pandas_dataset[
        "triple_barrier_event"
    ].tolist()
