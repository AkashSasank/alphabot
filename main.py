from dotenv import load_dotenv


from tradingbot.kite import (
    KiteSessionManager,
)

load_dotenv()
session_manager = KiteSessionManager()
session = session_manager.start_session(cli=False)
