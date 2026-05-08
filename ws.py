import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent / "packages" / "src"))

from tradingbot.kite import KiteSessionManager, KiteWebSocketClient

load_dotenv()
session = KiteSessionManager().start_session(cli=False)


def on_ticks(client, ticks):
    for tick in ticks:
        print(
            tick,
            flush=True,
        )


def on_connect(client, response):
    print("Connected to Kite websocket", response, flush=True)


def on_close(client, code, reason):
    print(f"Kite websocket closed: {code} - {reason}", flush=True)


def on_error(client, code, reason):
    print(f"Kite websocket error: {code} - {reason}", flush=True)


ws = KiteWebSocketClient(
    session=session,
    on_ticks=on_ticks,
    on_connect=on_connect,
    on_close=on_close,
    on_error=on_error,
    mode=KiteWebSocketClient.MODE_FULL,
)

try:
    ws.connect(symbols=["SBIN"], threaded=True)
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopping websocket...", flush=True)
    ws.close()
