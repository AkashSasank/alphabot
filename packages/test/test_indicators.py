from tradingbot.core.indicators import (
    AverageDirectionalIndex,
    ExponentialMovingAverageSlope,
    MACDHistogram,
    RollingVolatility,
    VolumeWeightedAveragePriceDistance,
    build_popular_indicators,
)
from tradingbot.indicators import (
    AverageDirectionalIndex as ExportedAverageDirectionalIndex,
    ExponentialMovingAverageSlope as ExportedExponentialMovingAverageSlope,
    MACDHistogram as ExportedMACDHistogram,
    RollingVolatility as ExportedRollingVolatility,
    VolumeWeightedAveragePriceDistance as ExportedVWAPDistance,
)
from tradingbot.visualization.constants import IndicatorUnit, IndicatorUnitRegistry


def _trend_candles(count: int = 60):
    closes = [
        100
        + (index * 0.7)
        + ((index % 5) - 2) * 0.4
        - (1.8 if index % 11 == 0 else 0)
        for index in range(count)
    ]
    return [
        {
            "timestamp": index,
            "open": close - 0.4,
            "high": close + 1.2 + ((index % 3) * 0.1),
            "low": close - 1.0 - ((index % 4) * 0.1),
            "close": close,
            "volume": 1000 + (index * 17),
        }
        for index, close in enumerate(closes)
    ]


def test_new_indicators_are_exported():
    assert ExportedAverageDirectionalIndex is AverageDirectionalIndex
    assert ExportedExponentialMovingAverageSlope is ExponentialMovingAverageSlope
    assert ExportedMACDHistogram is MACDHistogram
    assert ExportedRollingVolatility is RollingVolatility
    assert ExportedVWAPDistance is VolumeWeightedAveragePriceDistance


def test_new_indicators_compute_latest_values():
    candles = _trend_candles()
    indicators = [
        ExponentialMovingAverageSlope(period=9, slope_period=3),
        MACDHistogram(fast_period=12, slow_period=26, signal_period=9),
        AverageDirectionalIndex(period=14, normalize=True),
        VolumeWeightedAveragePriceDistance(period=20),
        RollingVolatility(period=20),
    ]

    for indicator in indicators:
        points = indicator.compute(candles)
        assert len(points) == len(candles)
        assert points[-1].timestamp == candles[-1]["timestamp"]
        assert points[-1].value is not None


def test_new_indicator_full_series_matches_prefix_latest():
    candles = _trend_candles()
    indicators = [
        ExponentialMovingAverageSlope(period=5, slope_period=2),
        MACDHistogram(fast_period=4, slow_period=8, signal_period=3),
        AverageDirectionalIndex(period=5, normalize=True),
        VolumeWeightedAveragePriceDistance(period=6),
        RollingVolatility(period=6),
    ]

    for indicator in indicators:
        full_series = indicator.compute(candles)
        prefix_series = [
            indicator.compute_point(candles[: index + 1])
            for index in range(len(candles))
        ]

        for full_point, prefix_point in zip(full_series, prefix_series):
            if full_point.value is None or prefix_point.value is None:
                assert full_point.value == prefix_point.value
            else:
                assert round(full_point.value, 10) == round(prefix_point.value, 10)


def test_normalized_directional_and_distance_indicators_are_bounded():
    candles = _trend_candles()
    indicators = [
        AverageDirectionalIndex(period=5, normalize=True),
        VolumeWeightedAveragePriceDistance(period=6),
    ]

    for indicator in indicators:
        values = [
            point.value
            for point in indicator.compute(candles)
            if point.value is not None
        ]
        assert values
        assert all(-1 <= value <= 1 for value in values)


def test_popular_indicators_include_trend_feature_set():
    indicator_names = {indicator.name for indicator in build_popular_indicators()}

    assert "ATR14" in indicator_names
    assert "RSI14" in indicator_names
    assert "EMASLOPE(9,3)" in indicator_names
    assert "MACDHIST(12,26,9)" in indicator_names
    assert "ADX14" in indicator_names
    assert "BBW(20,2.0)" in indicator_names
    assert "VWAPDIST20" in indicator_names
    assert "RVOL20" in indicator_names


def test_new_indicators_have_visualization_units():
    assert IndicatorUnitRegistry.unit_for("EMASLOPE(9,3)") == IndicatorUnit.PERCENT
    assert IndicatorUnitRegistry.unit_for("VWAPDIST20") == IndicatorUnit.PERCENT
    assert IndicatorUnitRegistry.unit_for("MACDHIST(12,26,9)") == IndicatorUnit.VOLATILITY
    assert IndicatorUnitRegistry.unit_for("ADX14") == IndicatorUnit.OSCILLATOR
    assert IndicatorUnitRegistry.unit_for("RVOL20") == IndicatorUnit.VOLATILITY
