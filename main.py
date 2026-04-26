from dotenv import load_dotenv
from tradingbot import Ticker
from tradingbot.core.candle_storage import FileCandelStorage
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
cache = FileCandelStorage(symbol="sbin", interval="1m")

ticker = Ticker(
    name="sbin",
    interval="1m",
    indicators=indicators,
    candle_api_provider=api,
    candle_cache=cache,
)
for i in range(10):
    tick = ticker.poll()
    print(tick.to_dict())
