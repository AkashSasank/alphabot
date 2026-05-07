import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent / "packages" / "src"))

from tradingbot import Ticker
from tradingbot.core.candle_storage import FileCandelStorage
from tradingbot.core.constants import Interval
from tradingbot.indicators import (
    MovingAverageConvergenceDivergence,
    SimpleMovingAverage,
)
from tradingbot.kite import KiteCandleAPIProvider, KiteSessionManager

load_dotenv()
session_manager = KiteSessionManager()
session = session_manager.start_session(cli=False)
api = KiteCandleAPIProvider(session=session)

indicators = [SimpleMovingAverage(10), MovingAverageConvergenceDivergence()]
timeframe = Interval.MINUTE
cache = FileCandelStorage(symbol="sbin")

ticker = Ticker(
    name="sbin",
    indicators=indicators,
    candle_api_provider=api,
    candle_cache=cache,
)
for i in range(10):
    tick = ticker.poll(timeframe=timeframe)
    # print(tick.to_dict())
