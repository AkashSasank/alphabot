import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from tradingbot.models import CandleWindowDataset, TransformerWindowDataset


def _write_csv(tmp_path, name, rows):
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_transformer_window_dataset_reads_one_hot_label_columns(tmp_path):
    path = _write_csv(
        tmp_path,
        "labels.csv",
        [
            {"f0": 0.0, "f1": 0.5, "lower": 0, "neutral": 0, "upper": 1},
            {"f0": 1.0, "f1": 1.5, "lower": 1, "neutral": 0, "upper": 0},
            {"f0": 2.0, "f1": 2.5, "lower": 0, "neutral": 1, "upper": 0},
            {"f0": 3.0, "f1": 3.5, "lower": 0, "neutral": 0, "upper": 1},
            {"f0": 4.0, "f1": 4.5, "lower": 1, "neutral": 0, "upper": 0},
        ],
    )

    dataset = TransformerWindowDataset(
        [path],
        feature_columns=["f0", "f1"],
        context_length=2,
        label_columns=["lower", "neutral", "upper"],
    )

    x, y = dataset[0]

    assert len(dataset) == 3
    assert x.shape == (2, 2)
    assert x.tolist() == [[0.0, 0.5], [1.0, 1.5]]
    assert y.item() == 1
    assert dataset.label_counts.tolist() == [1, 1, 1]


def test_transformer_window_dataset_can_return_one_hot_targets(tmp_path):
    path = _write_csv(
        tmp_path,
        "one_hot.csv",
        [
            {"f0": 0.0, "lower": 0, "neutral": 0, "upper": 1},
            {"f0": 1.0, "lower": 1, "neutral": 0, "upper": 0},
            {"f0": 2.0, "lower": 0, "neutral": 1, "upper": 0},
        ],
    )

    dataset = TransformerWindowDataset(
        [path],
        feature_columns=["f0"],
        context_length=2,
        label_columns=["lower", "neutral", "upper"],
        target_mode="one_hot",
    )

    _, y = dataset[0]

    assert y.tolist() == [0.0, 1.0, 0.0]


def test_transformer_window_dataset_maps_scalar_labels(tmp_path):
    path = _write_csv(
        tmp_path,
        "scalar.csv",
        [
            {"f0": 0.0, "label": 1},
            {"f0": 1.0, "label": -1},
            {"f0": 2.0, "label": 0},
            {"f0": 3.0, "label": 1},
        ],
    )

    dataset = TransformerWindowDataset(
        [path],
        feature_columns=["f0"],
        context_length=2,
        label_column="label",
        label_values=[-1, 0, 1],
    )

    assert dataset[0][1].item() == 1
    assert dataset[1][1].item() == 2


def test_transformer_window_dataset_can_return_regression_targets(tmp_path):
    path = _write_csv(
        tmp_path,
        "regression.csv",
        [
            {"f0": 0.0, "future_return": 0.01},
            {"f0": 1.0, "future_return": -0.02},
            {"f0": 2.0, "future_return": 0.03},
        ],
    )

    dataset = TransformerWindowDataset(
        [path],
        feature_columns=["f0"],
        context_length=2,
        target_columns=["future_return"],
        target_mode="regression",
    )

    assert dataset[0][1].tolist() == pytest.approx([0.03])


def test_candle_window_dataset_matches_gpt_notebook_constructor(tmp_path):
    path = _write_csv(
        tmp_path,
        "candle.csv",
        [
            {"f0": 0.0, "green": 1, "red": 0},
            {"f0": 1.0, "green": 0, "red": 1},
            {"f0": 2.0, "green": 1, "red": 0},
            {"f0": 3.0, "green": 0, "red": 1},
        ],
    )

    dataset = CandleWindowDataset(
        [path],
        feature_columns=["f0", "green", "red"],
        context_length=2,
        color_indices=[1, 2],
    )

    x, y = dataset[0]

    assert x.shape == (2, 3)
    assert y.item() == 0
    assert dataset.label_counts.tolist() == [1, 1]
