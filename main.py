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
    MovingAverageConvergenceDivergence,
    SimpleMovingAverage,
)
from tradingbot.kite import (
    KiteSessionManager,
)
from ticker import Ticker

SYMBOL = "SBIN"
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
ticker.add_indicator("sma_20", SimpleMovingAverage(period=20)).add_indicator(
    "sma_50", SimpleMovingAverage(period=50)
).add_indicator(
    "macd", MovingAverageConvergenceDivergence()
)
for i in range(100):
    candles = ticker.poll()
    print(candles)
    time.sleep(POLL_SECONDS)