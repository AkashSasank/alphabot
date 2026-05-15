from tradingbot.core.candles import Candle


def test_candle_to_candle_ratios_are_centered_around_zero():
    candle = Candle(
        timestamp="2026-05-12T09:15:00",
        open=98,
        high=110,
        low=90,
        close=102,
        volume=1000,
    )

    assert candle.properties.upper_wick_to_candle_ratio == -0.1
    assert candle.properties.lower_wick_to_candle_ratio == -0.1
    assert candle.properties.body_to_candle_ratio == -0.3
    assert candle.to_dict()["upper_wick_to_candle_ratio"] == -0.1
    assert candle.to_dict()["lower_wick_to_candle_ratio"] == -0.1
    assert candle.to_dict()["body_to_candle_ratio"] == -0.3


def test_centered_candle_ratios_stay_none_when_candle_size_is_zero():
    candle = Candle(
        timestamp="2026-05-12T09:15:00",
        open=100,
        high=100,
        low=100,
        close=100,
        volume=1000,
    )

    assert candle.properties.upper_wick_to_candle_ratio is None
    assert candle.properties.lower_wick_to_candle_ratio is None
    assert candle.properties.body_to_candle_ratio is None


def test_candle_color_is_green_when_lower_wick_is_larger():
    candle = Candle(
        timestamp="2026-05-14T09:15:00",
        open=102,
        high=108,
        low=90,
        close=106,
        volume=1000,
    )

    assert candle.properties.candle_color.value == "green"


def test_candle_color_is_red_when_lower_wick_is_not_larger():
    candle = Candle(
        timestamp="2026-05-14T09:15:00",
        open=100,
        high=112,
        low=98,
        close=104,
        volume=1000,
    )

    assert candle.properties.candle_color.value == "red"
