from collections import deque
from collections.abc import Iterable, Mapping
from math import isfinite, log1p
from statistics import stdev
from typing import Any

from tradingbot.core.candles import candle_builder
from tradingbot.core.constants import CandleColor, CandleType
from tradingbot.core.indicators import (
    AverageDirectionalIndex,
    AverageTrueRange,
    MovingAverageConvergenceDivergence,
    RelativeStrengthIndex,
    StochasticOscillator,
    StochasticRSI,
)


class CandleFeatureBuilder:
    CANDLE_COLOR_FEATURES = frozenset({"candle_color", "candle_color_code"})
    INDICATOR_FEATURES = frozenset(
        {"indicator", "indicators", "technical_indicators", "technical indicators"}
    )
    LOG_VOLUME_ZSCORE_FEATURE = "log_volume_zscore"
    ADX_FEATURES = frozenset({"adx"})
    ATR_FEATURES = frozenset({"atr"})
    MACD_FEATURES = frozenset({"macd"})
    RSI_FEATURES = frozenset({"rsi"})
    STOCHASTIC_FEATURES = frozenset(
        {"stochastic", "stoch", "stochastic_oscillator", "stochastic oscillator"}
    )
    STOCHASTIC_RSI_FEATURES = frozenset(
        {"stochastic_rsi", "stochastic rsi", "stoch_rsi", "stoch rsi", "stochrsi"}
    )
    WICK_RATIO_FEATURES = frozenset(
        {
            "wick_ratios",
            "upper_wick_to_candle_ratio",
            "lower_wick_to_candle_ratio",
            "body_to_candle_ratio",
        }
    )

    def __init__(
        self,
        feature_names: Iterable[str],
        *,
        log_volume_window: int = 5,
        log_volume_zscore_clip: float = 3.0,
        rsi_period: int = 14,
        rsi_periods: Iterable[int] | None = None,
        stochastic_period: int = 14,
        stochastic_periods: Iterable[int] | None = None,
        stochastic_rsi_period: int = 14,
        stochastic_rsi_stoch_period: int = 14,
        stochastic_rsi_periods: Iterable[int] | None = None,
        macd_fast_period: int = 12,
        macd_slow_period: int = 26,
        macd_close_ratio_clip: float = 0.05,
        atr_period: int = 14,
        atr_close_ratio_clip: float = 0.05,
        adx_period: int = 14,
    ):
        self.feature_names = list(feature_names)
        self.log_volume_window = log_volume_window
        self.log_volume_zscore_clip = log_volume_zscore_clip
        self.rsi_period = rsi_period
        self.rsi_periods = self._normalize_periods(rsi_periods, default=(7, 14))
        self.stochastic_period = stochastic_period
        self.stochastic_periods = self._normalize_periods(
            stochastic_periods,
            default=(7, 14),
        )
        self.stochastic_rsi_period = stochastic_rsi_period
        self.stochastic_rsi_stoch_period = stochastic_rsi_stoch_period
        self.stochastic_rsi_periods = self._normalize_periods(
            stochastic_rsi_periods,
            default=(7, 14),
        )
        self.macd_fast_period = macd_fast_period
        self.macd_slow_period = macd_slow_period
        self.macd_close_ratio_clip = macd_close_ratio_clip
        self.atr_period = atr_period
        self.atr_close_ratio_clip = atr_close_ratio_clip
        self.adx_period = adx_period
        self._log_volume_window = deque(maxlen=log_volume_window)
        self._indicator_candles: list[dict[str, Any]] = []
        self._rsi_indicator = RelativeStrengthIndex(
            period=rsi_period,
            normalize=True,
        )
        self._bundle_rsi_indicators = {
            period: RelativeStrengthIndex(period=period, normalize=True)
            for period in self.rsi_periods
        }
        self._stochastic_indicator = StochasticOscillator(
            period=stochastic_period,
            normalize=True,
        )
        self._bundle_stochastic_indicators = {
            period: StochasticOscillator(period=period, normalize=True)
            for period in self.stochastic_periods
        }
        self._stochastic_rsi_indicator = StochasticRSI(
            rsi_period=stochastic_rsi_period,
            stoch_period=stochastic_rsi_stoch_period,
            normalize=True,
        )
        self._bundle_stochastic_rsi_indicators = {
            period: StochasticRSI(
                rsi_period=period,
                stoch_period=period,
                normalize=True,
            )
            for period in self.stochastic_rsi_periods
        }
        self._macd_indicator = MovingAverageConvergenceDivergence(
            fast_period=macd_fast_period,
            slow_period=macd_slow_period,
        )
        self._atr_indicator = AverageTrueRange(period=atr_period)
        self._adx_indicator = AverageDirectionalIndex(
            period=adx_period,
            normalize=True,
        )
        self.output_columns = self._determine_output_columns()
        self.feature_map = {
            feature_name: index
            for index, feature_name in enumerate(self.output_columns)
        }

    def encode(self, candle: Mapping[str, Any]) -> dict[str, Any]:
        features = {}
        volume = float(candle["volume"])
        close = float(candle["close"])
        indicator_candle = {
            "timestamp": candle["timestamp"],
            "open": float(candle["open"]),
            "high": float(candle["high"]),
            "low": float(candle["low"]),
            "close": close,
            "volume": volume,
        }
        if self._has_indicator_feature():
            self._indicator_candles.append(indicator_candle)

        candle_ = candle_builder.build_candle(
            open=candle["open"],
            high=candle["high"],
            low=candle["low"],
            close=close,
            volume=volume,
            timestamp=candle["timestamp"],
        )
        candle_properties = candle_.properties
        for feature_name in self.feature_names:
            if feature_name == "timestamp":
                features["timestamp"] = candle["timestamp"]
            elif feature_name in self.CANDLE_COLOR_FEATURES:
                features.update(candle_properties.candle_color.one_hot_encode())
            elif feature_name == "candle_type":
                features.update(candle_properties.candle_type.one_hot_encode())
            elif feature_name in self.WICK_RATIO_FEATURES:
                features["upper_wick_to_candle_ratio"] = (
                    candle_properties.upper_wick_to_candle_ratio
                )
                features["lower_wick_to_candle_ratio"] = (
                    candle_properties.lower_wick_to_candle_ratio
                )
                features["body_to_candle_ratio"] = (
                    candle_properties.body_to_candle_ratio
                )
            elif feature_name == self.LOG_VOLUME_ZSCORE_FEATURE:
                features[self.LOG_VOLUME_ZSCORE_FEATURE] = (
                    self._iterative_log_volume_zscore(volume)
                )
            elif feature_name in self.INDICATOR_FEATURES:
                features.update(self._encode_indicator_bundle(close))
            elif feature_name in self.RSI_FEATURES:
                features["rsi"] = self._latest_rsi()
            elif feature_name in self.STOCHASTIC_FEATURES:
                features["stochastic"] = self._latest_stochastic()
            elif feature_name in self.STOCHASTIC_RSI_FEATURES:
                features["stochastic_rsi"] = self._latest_stochastic_rsi()
            elif feature_name in self.MACD_FEATURES:
                features["macd"] = self._latest_macd(close)
            elif feature_name in self.ATR_FEATURES:
                features["atr"] = self._latest_atr(close)
            elif feature_name in self.ADX_FEATURES:
                features["adx"] = self._latest_adx()
        return {column: features.get(column) for column in self.output_columns}

    def encode_vector(self, candle: Mapping[str, Any]) -> list[Any]:
        """Encode one candle as a positional feature vector."""
        encoded = self.encode(candle)
        return [encoded[column] for column in self.output_columns]

    def encode_dataframe(self, candles: Any) -> Any:
        """Encode a pandas-like candle dataframe into feature columns.

        This is the batch counterpart to ``encode``. It keeps the same output
        contract but avoids per-row Series construction in Dask partitions.
        """
        import pandas as pd

        output = pd.DataFrame(index=candles.index)

        open_ = candles["open"].astype(float)
        high = candles["high"].astype(float)
        low = candles["low"].astype(float)
        close = candles["close"].astype(float)
        volume = candles["volume"].astype(float)

        body_size = (close - open_).abs()
        candle_size = (high - low).clip(lower=0.0)
        upper_wick_size = (high - pd.concat([open_, close], axis=1).max(axis=1)).clip(
            lower=0.0
        )
        lower_wick_size = (pd.concat([open_, close], axis=1).min(axis=1) - low).clip(
            lower=0.0
        )

        if "timestamp" in self.feature_names:
            output["timestamp"] = candles["timestamp"]

        if self._has_any(self.CANDLE_COLOR_FEATURES):
            color = self._infer_color_series(upper_wick_size, lower_wick_size)
            for candle_color in CandleColor:
                output[f"candle_color_{candle_color.value}"] = (
                    color == candle_color.value
                ).astype(int)

        if "candle_type" in self.feature_names:
            candle_type = self._infer_type_series(
                body_size,
                candle_size,
                upper_wick_size,
                lower_wick_size,
            )
            for ctype in CandleType:
                output[f"candle_type_{ctype.value}"] = (
                    candle_type == ctype.value
                ).astype(int)
        if self._has_any(self.WICK_RATIO_FEATURES):
            output["upper_wick_to_candle_ratio"] = self._centered_ratio_series(
                upper_wick_size,
                candle_size,
            )
            output["lower_wick_to_candle_ratio"] = self._centered_ratio_series(
                lower_wick_size,
                candle_size,
            )
            output["body_to_candle_ratio"] = self._centered_ratio_series(
                body_size,
                candle_size,
            )

        if self.LOG_VOLUME_ZSCORE_FEATURE in self.feature_names:
            output[self.LOG_VOLUME_ZSCORE_FEATURE] = self._log_volume_zscore_series(
                volume
            )

        if self._has_indicator_feature():
            indicator_candles = None

            if self._has_full_indicator_bundle():
                for period in self.rsi_periods:
                    output[self._period_column("rsi", period)] = (
                        RelativeStrengthIndex.compute_dataframe(
                            close,
                            period=period,
                            normalize=True,
                        ).fillna(0.0)
                    )

                for period in self.stochastic_periods:
                    output[self._period_column("stochastic", period)] = (
                        StochasticOscillator.compute_dataframe(
                            high,
                            low,
                            close,
                            period=period,
                            normalize=True,
                        ).fillna(0.0)
                    )

                for period in self.stochastic_rsi_periods:
                    output[self._period_column("stochastic_rsi", period)] = (
                        StochasticRSI.compute_dataframe(
                            close,
                            rsi_period=period,
                            stoch_period=period,
                            normalize=True,
                        ).fillna(0.0)
                    )

            if self._has_rsi_feature():
                output["rsi"] = RelativeStrengthIndex.compute_dataframe(
                    close,
                    period=self.rsi_period,
                    normalize=True,
                ).fillna(0.0)

            if self._has_stochastic_feature():
                output["stochastic"] = StochasticOscillator.compute_dataframe(
                    high,
                    low,
                    close,
                    period=self.stochastic_period,
                    normalize=True,
                ).fillna(0.0)

            if self._has_stochastic_rsi_feature():
                output["stochastic_rsi"] = StochasticRSI.compute_dataframe(
                    close,
                    rsi_period=self.stochastic_rsi_period,
                    stoch_period=self.stochastic_rsi_stoch_period,
                    normalize=True,
                ).fillna(0.0)

            if self._has_macd_feature() or self._has_adx_feature():
                indicator_candles = candles[
                    ["timestamp", "open", "high", "low", "close", "volume"]
                ].to_dict("records")

            if self._has_macd_feature():
                macd = self._indicator_series(
                    MovingAverageConvergenceDivergence(
                        fast_period=self.macd_fast_period,
                        slow_period=self.macd_slow_period,
                    ),
                    indicator_candles or [],
                    candles.index,
                )
                output["macd"] = self._normalize_macd_series(macd, close)

            if self._has_atr_feature():
                atr = AverageTrueRange.compute_dataframe(
                    high,
                    low,
                    close,
                    period=self.atr_period,
                ).fillna(0.0)
                output["atr"] = self._normalize_atr_series(atr, close)

            if self._has_adx_feature():
                output["adx"] = self._indicator_series(
                    AverageDirectionalIndex(
                        period=self.adx_period,
                        normalize=True,
                    ),
                    indicator_candles or [],
                    candles.index,
                )

        return output.reindex(columns=self.output_columns)

    def encode_dataframe_values(self, candles: Any) -> Any:
        """Encode dataframe candles and return values in feature-map order."""
        return self.encode_dataframe(candles).to_numpy()

    def reset_state(self) -> None:
        """Clear rolling state used by iterative single-candle features."""
        self._log_volume_window.clear()
        self._indicator_candles.clear()

    def _determine_output_columns(self) -> list[str]:
        columns = []
        for feature_name in self.feature_names:
            if feature_name == "timestamp":
                columns.append("timestamp")
            elif feature_name in self.CANDLE_COLOR_FEATURES:
                columns.extend(f"candle_color_{color.value}" for color in CandleColor)
            elif feature_name == "candle_type":
                columns.extend(f"candle_type_{ctype.value}" for ctype in CandleType)
            elif feature_name in self.WICK_RATIO_FEATURES:
                columns.extend(
                    [
                        "upper_wick_to_candle_ratio",
                        "lower_wick_to_candle_ratio",
                        "body_to_candle_ratio",
                    ]
                )
            elif feature_name == self.LOG_VOLUME_ZSCORE_FEATURE:
                columns.append(self.LOG_VOLUME_ZSCORE_FEATURE)
            elif feature_name in self.INDICATOR_FEATURES:
                columns.extend(self._indicator_output_columns())
            elif feature_name in self.RSI_FEATURES:
                columns.append("rsi")
            elif feature_name in self.STOCHASTIC_FEATURES:
                columns.append("stochastic")
            elif feature_name in self.STOCHASTIC_RSI_FEATURES:
                columns.append("stochastic_rsi")
            elif feature_name in self.MACD_FEATURES:
                columns.append("macd")
            elif feature_name in self.ATR_FEATURES:
                columns.append("atr")
            elif feature_name in self.ADX_FEATURES:
                columns.append("adx")
            else:
                columns.append(feature_name)
        return list(dict.fromkeys(columns))

    def _has_any(self, feature_names: frozenset[str]) -> bool:
        return any(feature_name in self.feature_names for feature_name in feature_names)

    def _has_indicator_feature(self) -> bool:
        return (
            self._has_full_indicator_bundle()
            or self._has_rsi_feature()
            or self._has_stochastic_feature()
            or self._has_stochastic_rsi_feature()
            or self._has_macd_feature()
            or self._has_atr_feature()
            or self._has_adx_feature()
        )

    def _has_rsi_feature(self) -> bool:
        return self._has_any(self.RSI_FEATURES)

    def _has_stochastic_feature(self) -> bool:
        return self._has_any(self.STOCHASTIC_FEATURES)

    def _has_stochastic_rsi_feature(self) -> bool:
        return self._has_any(self.STOCHASTIC_RSI_FEATURES)

    def _has_macd_feature(self) -> bool:
        return self._has_full_indicator_bundle() or self._has_any(self.MACD_FEATURES)

    def _has_atr_feature(self) -> bool:
        return self._has_full_indicator_bundle() or self._has_any(self.ATR_FEATURES)

    def _has_adx_feature(self) -> bool:
        return self._has_full_indicator_bundle() or self._has_any(self.ADX_FEATURES)

    def _has_full_indicator_bundle(self) -> bool:
        return self._has_any(self.INDICATOR_FEATURES)

    def _indicator_output_columns(self) -> list[str]:
        columns = []
        columns.extend(self._period_column("rsi", period) for period in self.rsi_periods)
        columns.extend(
            self._period_column("stochastic", period)
            for period in self.stochastic_periods
        )
        columns.extend(
            self._period_column("stochastic_rsi", period)
            for period in self.stochastic_rsi_periods
        )
        columns.extend(["macd", "atr", "adx"])
        return columns

    @staticmethod
    def _period_column(name: str, period: int) -> str:
        return f"{name}_{period}"

    @staticmethod
    def _normalize_periods(
        periods: Iterable[int] | None,
        *,
        default: tuple[int, ...],
    ) -> tuple[int, ...]:
        resolved = tuple(default if periods is None else periods)
        if not resolved:
            raise ValueError("periods must not be empty")
        if any(period <= 0 for period in resolved):
            raise ValueError("periods must be greater than 0")
        return tuple(dict.fromkeys(resolved))

    @staticmethod
    def _centered_ratio_series(numerator: Any, denominator: Any) -> Any:
        return numerator.div(denominator.where(denominator != 0)) - 0.5

    def _iterative_log_volume_zscore(self, volume: float) -> float | None:
        log_volume = log1p(max(volume, 0.0))
        if len(self._log_volume_window) < self.log_volume_window:
            self._log_volume_window.append(log_volume)
            return None

        mean = sum(self._log_volume_window) / self.log_volume_window
        std = stdev(self._log_volume_window)
        self._log_volume_window.append(log_volume)
        if std == 0:
            return 0.0

        return self._bounded_zscore((log_volume - mean) / std)

    def _log_volume_zscore_series(self, volume: Any) -> Any:
        import numpy as np

        log_volume = np.log1p(volume.where(volume >= 0))
        rolling = log_volume.rolling(
            window=self.log_volume_window,
            min_periods=self.log_volume_window,
        )
        rolling_mean = rolling.mean().shift(1)
        rolling_std = rolling.std().shift(1)
        zscore = (log_volume - rolling_mean) / rolling_std
        return zscore.fillna(0.0).map(self._bounded_zscore)

    def _bounded_zscore(self, zscore: float) -> float:
        return max(
            -self.log_volume_zscore_clip,
            min(self.log_volume_zscore_clip, zscore),
        ) / self.log_volume_zscore_clip

    def _encode_indicator_bundle(self, close: float) -> dict[str, float | None]:
        features = {
            "macd": self._latest_macd(close),
            "atr": self._latest_atr(close),
            "adx": self._latest_adx(),
        }
        features.update(
            {
                self._period_column("rsi", period): indicator.compute_point(
                    self._indicator_candles
                ).value
                for period, indicator in self._bundle_rsi_indicators.items()
            }
        )
        features.update(
            {
                self._period_column("stochastic", period): indicator.compute_point(
                    self._indicator_candles
                ).value
                for period, indicator in self._bundle_stochastic_indicators.items()
            }
        )
        features.update(
            {
                self._period_column(
                    "stochastic_rsi",
                    period,
                ): indicator.compute_point(self._indicator_candles).value
                for period, indicator in self._bundle_stochastic_rsi_indicators.items()
            }
        )
        return features

    def _latest_rsi(self) -> float | None:
        return self._rsi_indicator.compute_point(self._indicator_candles).value

    def _latest_stochastic(self) -> float | None:
        return self._stochastic_indicator.compute_point(self._indicator_candles).value

    def _latest_stochastic_rsi(self) -> float | None:
        return self._stochastic_rsi_indicator.compute_point(
            self._indicator_candles
        ).value

    def _latest_macd(self, close: float) -> float | None:
        point = self._macd_indicator.compute_point(self._indicator_candles)
        return self._normalize_macd_value(point.value, close)

    def _latest_atr(self, close: float) -> float | None:
        point = self._atr_indicator.compute_point(self._indicator_candles)
        return self._normalize_atr_value(point.value, close)

    def _latest_adx(self) -> float | None:
        return self._adx_indicator.compute_point(self._indicator_candles).value

    @staticmethod
    def _indicator_series(
        indicator: Any,
        candles: list[dict[str, Any]],
        index: Any,
    ) -> Any:
        import pandas as pd

        if not candles:
            return pd.Series(index=index, dtype="float64")

        values = [point.value for point in indicator.compute(candles)]
        return pd.Series(values, index=index, dtype="float64").fillna(0.0)

    def _normalize_macd_series(self, macd: Any, close: Any) -> Any:
        close_ratio = macd.div(close.where(close != 0))
        return close_ratio.fillna(0.0).map(self._bounded_macd_close_ratio)

    def _normalize_macd_value(self, macd: float | None, close: float) -> float | None:
        if macd is None:
            return None
        if close == 0:
            return 0.0
        return self._bounded_macd_close_ratio(macd / close)

    def _bounded_macd_close_ratio(self, ratio: float) -> float:
        if not isfinite(ratio):
            return 0.0
        return max(
            -self.macd_close_ratio_clip,
            min(self.macd_close_ratio_clip, ratio),
        ) / self.macd_close_ratio_clip

    def _normalize_atr_series(self, atr: Any, close: Any) -> Any:
        close_ratio = atr.div(close.where(close != 0))
        return close_ratio.fillna(0.0).map(self._bounded_atr_close_ratio)

    def _normalize_atr_value(self, atr: float | None, close: float) -> float | None:
        if atr is None:
            return None
        if close == 0:
            return 0.0
        return self._bounded_atr_close_ratio(atr / close)

    def _bounded_atr_close_ratio(self, ratio: float) -> float:
        if not isfinite(ratio):
            return 0.0
        return max(
            0.0,
            min(self.atr_close_ratio_clip, ratio),
        ) / self.atr_close_ratio_clip

    @staticmethod
    def _infer_color_series(upper_wick_size: Any, lower_wick_size: Any) -> Any:
        import pandas as pd

        color = pd.Series(CandleColor.RED.value, index=upper_wick_size.index)
        color.loc[lower_wick_size > upper_wick_size] = CandleColor.GREEN.value
        return color

    @staticmethod
    def _infer_type_series(
        body_size: Any,
        candle_size: Any,
        upper_wick_size: Any,
        lower_wick_size: Any,
    ) -> Any:
        import pandas as pd

        candle_type = pd.Series(CandleType.STANDARD.value, index=body_size.index)

        doji = (candle_size == 0) | (body_size <= candle_size * 0.1)
        candle_type.loc[doji] = CandleType.DOJI.value

        unclassified = ~doji
        marubozu = unclassified & (body_size >= candle_size * 0.9)
        candle_type.loc[marubozu] = CandleType.MARUBOZU.value

        unclassified = unclassified & ~marubozu
        hammer = unclassified & (
            (lower_wick_size >= body_size * 2) & (upper_wick_size <= body_size)
        )
        candle_type.loc[hammer] = CandleType.HAMMER.value

        unclassified = unclassified & ~hammer
        inverted_hammer = unclassified & (
            (upper_wick_size >= body_size * 2) & (lower_wick_size <= body_size)
        )
        candle_type.loc[inverted_hammer] = CandleType.INVERTED_HAMMER.value

        unclassified = unclassified & ~inverted_hammer
        spinning_top = unclassified & (body_size <= candle_size * 0.3)
        candle_type.loc[spinning_top] = CandleType.SPINNING_TOP.value

        return candle_type
