import pandas as pd

from tradingbot.core.features import CandleFeatureBuilder


FEATURES = [
    "timestamp",
    "upper_wick_to_candle_ratio",
    "lower_wick_to_candle_ratio",
    "body_to_candle_ratio",
    "candle_type",
    "candle_color_code",
    "log_volume_zscore",
]


def test_candle_feature_builder_outputs_binary_color_columns():
    builder = CandleFeatureBuilder(FEATURES)

    assert builder.output_columns == [
        "timestamp",
        "upper_wick_to_candle_ratio",
        "lower_wick_to_candle_ratio",
        "body_to_candle_ratio",
        "candle_type_standard",
        "candle_type_doji",
        "candle_type_hammer",
        "candle_type_inverted_hammer",
        "candle_type_spinning_top",
        "candle_type_marubozu",
        "candle_color_green",
        "candle_color_red",
        "log_volume_zscore",
    ]
    assert builder.feature_map == {
        "timestamp": 0,
        "upper_wick_to_candle_ratio": 1,
        "lower_wick_to_candle_ratio": 2,
        "body_to_candle_ratio": 3,
        "candle_type_standard": 4,
        "candle_type_doji": 5,
        "candle_type_hammer": 6,
        "candle_type_inverted_hammer": 7,
        "candle_type_spinning_top": 8,
        "candle_type_marubozu": 9,
        "candle_color_green": 10,
        "candle_color_red": 11,
        "log_volume_zscore": 12,
    }


def test_single_candle_encoding_uses_lower_wick_pressure_for_green():
    builder = CandleFeatureBuilder(FEATURES)
    encoded = builder.encode(
        {
            "timestamp": "2026-05-14T09:15:00+05:30",
            "open": 102,
            "high": 108,
            "low": 90,
            "close": 106,
            "volume": 1000,
        }
    )

    assert encoded["candle_color_green"] == 1
    assert encoded["candle_color_red"] == 0
    assert encoded["candle_type_hammer"] == 1


def test_single_candle_encoding_uses_red_when_lower_wick_is_not_larger():
    builder = CandleFeatureBuilder(FEATURES)
    encoded = builder.encode(
        {
            "timestamp": "2026-05-14T09:15:00+05:30",
            "open": 100,
            "high": 112,
            "low": 98,
            "close": 104,
            "volume": 1000,
        }
    )

    assert encoded["candle_color_green"] == 0
    assert encoded["candle_color_red"] == 1


def test_dataframe_encoding_matches_single_candle_encoding():
    builder = CandleFeatureBuilder(
        [feature for feature in FEATURES if feature != "log_volume_zscore"]
    )
    candle = {
        "timestamp": "2026-05-14T09:15:00+05:30",
        "open": 102,
        "high": 108,
        "low": 90,
        "close": 106,
        "volume": 1000,
    }

    single = builder.encode(candle)
    dataframe_row = builder.encode_dataframe(pd.DataFrame([candle])).iloc[0].to_dict()

    assert dataframe_row.keys() == single.keys()
    for key, value in single.items():
        if key == "timestamp":
            assert dataframe_row[key] == value
        else:
            assert round(float(dataframe_row[key]), 10) == round(float(value), 10)


def test_encode_vector_uses_feature_map_order():
    builder = CandleFeatureBuilder(
        [feature for feature in FEATURES if feature != "log_volume_zscore"]
    )
    candle = {
        "timestamp": "2026-05-14T09:15:00+05:30",
        "open": 102,
        "high": 108,
        "low": 90,
        "close": 106,
        "volume": 1000,
    }

    encoded = builder.encode(candle)
    vector = builder.encode_vector(candle)

    for feature_name, index in builder.feature_map.items():
        assert vector[index] == encoded[feature_name]


def test_dataframe_encoding_adds_bounded_log_volume_zscore():
    builder = CandleFeatureBuilder(FEATURES, log_volume_window=5)
    candles = pd.DataFrame(
        [
            {
                "timestamp": index,
                "open": 100,
                "high": 110,
                "low": 90,
                "close": 105,
                "volume": volume,
            }
            for index, volume in enumerate([100, 110, 120, 130, 140, 1000, 90])
        ]
    )

    encoded = builder.encode_dataframe(candles)

    assert encoded["log_volume_zscore"].between(-1, 1).all()
    assert encoded["log_volume_zscore"].iloc[:5].tolist() == [0, 0, 0, 0, 0]
    assert encoded["log_volume_zscore"].iloc[5] > 0


def test_iterative_log_volume_zscore_matches_dataframe_encoding_after_warmup():
    volumes = [100, 110, 120, 130, 140, 1000, 90]
    candles = [
        {
            "timestamp": index,
            "open": 100,
            "high": 110,
            "low": 90,
            "close": 105,
            "volume": volume,
        }
        for index, volume in enumerate(volumes)
    ]
    iterative_builder = CandleFeatureBuilder(FEATURES, log_volume_window=5)
    dataframe_builder = CandleFeatureBuilder(FEATURES, log_volume_window=5)

    iterative_values = [
        iterative_builder.encode(candle)["log_volume_zscore"] for candle in candles
    ]
    dataframe_values = dataframe_builder.encode_dataframe(pd.DataFrame(candles))[
        "log_volume_zscore"
    ].tolist()

    assert iterative_values[:5] == [None, None, None, None, None]
    for iterative, dataframe in zip(iterative_values[5:], dataframe_values[5:]):
        assert round(iterative, 10) == round(dataframe, 10)


def _indicator_test_candles():
    closes = [
        100,
        102,
        101,
        104,
        103,
        105,
        102,
        106,
        108,
        107,
        109,
        106,
        110,
        112,
        111,
    ]
    return [
        {
            "timestamp": index,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.5,
            "close": close,
            "volume": 1000 + (index * 10),
        }
        for index, close in enumerate(closes)
    ]


def test_indicator_bundle_adds_normalized_output_columns():
    builder = CandleFeatureBuilder(["timestamp", "indicators"])

    assert builder.output_columns == [
        "timestamp",
        "rsi",
        "stochastic_rsi",
        "macd",
        "atr",
        "adx",
    ]
    assert builder.feature_map == {
        "timestamp": 0,
        "rsi": 1,
        "stochastic_rsi": 2,
        "macd": 3,
        "atr": 4,
        "adx": 5,
    }


def test_dataframe_indicator_features_are_normalized():
    builder = CandleFeatureBuilder(
        ["indicators"],
        rsi_period=3,
        stochastic_rsi_period=3,
        stochastic_rsi_stoch_period=3,
        macd_fast_period=3,
        macd_slow_period=5,
        atr_period=3,
        adx_period=3,
    )

    encoded = builder.encode_dataframe(pd.DataFrame(_indicator_test_candles()))

    assert encoded.columns.tolist() == ["rsi", "stochastic_rsi", "macd", "atr", "adx"]
    for column in encoded.columns:
        assert encoded[column].between(-1, 1).all()
    assert encoded.iloc[-1].notna().all()


def test_iterative_indicator_features_match_dataframe_after_warmup():
    candles = _indicator_test_candles()
    iterative_builder = CandleFeatureBuilder(
        ["indicators"],
        rsi_period=3,
        stochastic_rsi_period=3,
        stochastic_rsi_stoch_period=3,
        macd_fast_period=3,
        macd_slow_period=5,
        atr_period=3,
        adx_period=3,
    )
    dataframe_builder = CandleFeatureBuilder(
        ["indicators"],
        rsi_period=3,
        stochastic_rsi_period=3,
        stochastic_rsi_stoch_period=3,
        macd_fast_period=3,
        macd_slow_period=5,
        atr_period=3,
        adx_period=3,
    )

    iterative_last = [iterative_builder.encode(candle) for candle in candles][-1]
    dataframe_last = dataframe_builder.encode_dataframe(pd.DataFrame(candles)).iloc[
        -1
    ].to_dict()

    for column in ["rsi", "stochastic_rsi", "macd", "atr", "adx"]:
        assert round(iterative_last[column], 10) == round(dataframe_last[column], 10)


def test_indicator_alias_adds_full_indicator_bundle():
    builder = CandleFeatureBuilder(["indicator"])

    assert builder.output_columns == ["rsi", "stochastic_rsi", "macd", "atr", "adx"]


def test_empty_dataframe_indicator_features_return_schema_without_computing():
    builder = CandleFeatureBuilder(["timestamp", "indicators"])
    empty = pd.DataFrame(
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )

    encoded = builder.encode_dataframe(empty)

    assert encoded.empty
    assert encoded.columns.tolist() == [
        "timestamp",
        "rsi",
        "stochastic_rsi",
        "macd",
        "atr",
        "adx",
    ]
