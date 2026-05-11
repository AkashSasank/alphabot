import sys
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time

sys.path.insert(0, str(Path(__file__).resolve().parent / "packages" / "src"))

from tradingbot.core.sequence import sequence_builder
from tradingbot.core.candles import candle_builder
from tradingbot.core.constants import Interval
from tradingbot.indicators import (
    RelativeStrengthIndex,
    StochasticOscillator,
    StochasticRSI,
)
from tradingbot.kite import (
    KiteSessionManager,
)
from tradingbot.core.ticker import Ticker

STRONG_STOCKS = [
    "RELIANCE",
    "HDFCBANK",
    "ICICIBANK",
    "INFY",
    "TCS",
    "SBIN",
    "AXISBANK",
    "KOTAKBANK",
    "LT",
    "ITC",
    "HINDUNILVR",
    "BHARTIARTL",
    "MARUTI",
    "M&M",
    "SUNPHARMA",
    "HCLTECH",
    "TECHM",
    "WIPRO",
    "TITAN",
    "ASIANPAINT",
    "BAJFINANCE",
    "BAJAJFINSV",
    "ULTRACEMCO",
    "NTPC",
    "POWERGRID",
    "ONGC",
    "COALINDIA",
    "TATASTEEL",
    "JSWSTEEL",
    "HINDALCO",
    "GRASIM",
    "ADANIPORTS",
    "DRREDDY",
    "CIPLA",
    "EICHERMOT",
    "HEROMOTOCO",
    "BAJAJ-AUTO",
    "NESTLEIND",
]
SYMBOL = STRONG_STOCKS[0]
INTERVAL = Interval.MINUTE
SEQUENCE_CAPACITY = 90
RECENT_CONSISTENCY_WINDOW = 10
POLL_SECONDS = 3
repair_state = {
    "last_repair_bucket": None,
    "last_gap_key": None,
}


load_dotenv()
session_manager = KiteSessionManager()
session = session_manager.start_session(cli=False)

ticker = Ticker(symbol=SYMBOL, timeframe="1m", session=session)
ticker.add_indicator(
    "rsi_7", RelativeStrengthIndex(period=7, normalize=True)
).add_indicator(
    "rsi_14", RelativeStrengthIndex(period=14, normalize=True)
).add_indicator(
    "stoch_14", StochasticOscillator(period=14, normalize=True)
).add_indicator(
    "stoch_21", StochasticOscillator(period=21, normalize=True)
).add_indicator(
    "stoch_rsi_14_14",
    StochasticRSI(rsi_period=14, stoch_period=14, normalize=True),
)
# for i in range(100):
#     candles = ticker.poll()
#     print(candles)
#     time.sleep(POLL_SECONDS)
path = ticker.get_historic_data(
    ticker_name="SBIN",
    timeframe="5m",
)
ticker.update_historic_data_indicators(
path
)
