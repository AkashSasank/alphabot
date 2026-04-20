"""Minimal BotManager workflow that renders one polled chart snapshot."""

import os
import shutil
import subprocess
from datetime import datetime, timedelta

from dotenv import load_dotenv
from tradingbot.bot import BotManager
from tradingbot.core.indicators import (
    AverageTrueRange,
    ExponentialMovingAverage,
    RelativeStrengthIndex,
    VolumeWeightedAveragePrice,
)
from tradingbot.visualization import GradioChartApp, RealtimeTradingChart

try:
    from PIL import Image
except ImportError:
    Image = None


def _build_manager() -> BotManager:
    """Create a configured bot manager for demo and manual runs."""
    config = {
        "api_key": os.getenv("KITE_API_KEY"),
        "api_secret": os.getenv("KITE_API_SECRET"),
        "user_id": os.getenv("KITE_USER_ID"),
        "password": os.getenv("KITE_PASSWORD"),
        "pin": os.getenv("KITE_PIN"),
        "headless": os.getenv("KITE_HEADLESS", "False").lower() == "true",
        "timeout_ms": int(os.getenv("KITE_TIMEOUT_MS", "45000")),
        "redirect_url": os.getenv(
            "KITE_REDIRECT_URL",
            "http://localhost:1130/",
        ),
    }

    manager = BotManager(config)

    indicators = [
        ExponentialMovingAverage(period=9),
        ExponentialMovingAverage(period=21),
        RelativeStrengthIndex(period=14),
        AverageTrueRange(period=14),
        VolumeWeightedAveragePrice(period=10),
    ]

    manager.add_bot_and_subscribe(
        ticker_name="SBIN",
        interval="5m",
        indicators=indicators,
        api_limit=100,
    )

    return manager


def _recent_session_timestamps(days_back: int = 7) -> list[datetime]:
    """Return recent weekday session timestamps for fallback polling."""
    now = datetime.now()
    timestamps: list[datetime] = []

    for day_offset in range(days_back + 1):
        candidate_day = now - timedelta(days=day_offset)
        if candidate_day.weekday() >= 5:
            continue

        timestamps.append(
            candidate_day.replace(
                hour=15,
                minute=20,
                second=0,
                microsecond=0,
            )
        )

    return timestamps


def _poll_with_fallback(manager: BotManager, bot_id: str):
    """Poll latest state and retry using recent session dates if empty."""
    polled = manager.poll_bot(
        bot_id=bot_id,
        limit=None,
        emit_updates=False,
    )

    if len(polled.candles) == 0:
        for fallback_time in _recent_session_timestamps(days_back=10):
            polled = manager.poll_bot(
                bot_id=bot_id,
                limit=None,
                emit_updates=False,
                date=fallback_time,
            )
            if len(polled.candles) > 0:
                print(f"Fallback poll succeeded with date={fallback_time}")
                break

    return polled


def _save_and_open_chart(
    chart: RealtimeTradingChart,
    snapshot_name: str,
) -> None:
    """Persist the current chart and open it in Preview on macOS."""
    snapshot_path = os.path.join(os.getcwd(), snapshot_name)
    if chart.figure is not None:
        chart.figure.savefig(snapshot_path, dpi=150)
        subprocess.run(["open", snapshot_path], check=False)
        print(f"Chart snapshot opened: {snapshot_path}")
    else:
        print("No figure generated; unable to open chart snapshot.")


def _launch_gradio_chart(
    title: str,
    payload,
) -> None:
    """Launch the current payload in a full-window Gradio browser app."""
    app = GradioChartApp(title=title)
    app.launch(payload)


def _prepare_backtest_frames_dir() -> str:
    """Create a clean directory for step-by-step backtest chart frames."""
    frames_dir = os.path.join(os.getcwd(), "backtest_frames")
    shutil.rmtree(frames_dir, ignore_errors=True)
    os.makedirs(frames_dir, exist_ok=True)
    return frames_dir


def _save_backtest_frame(
    chart: RealtimeTradingChart,
    frames_dir: str,
    step_index: int,
) -> str | None:
    """Persist one backtest chart frame to disk."""
    if chart.figure is None:
        return None

    frame_path = os.path.join(frames_dir, f"frame_{step_index:04d}.png")
    chart.figure.savefig(frame_path, dpi=150)
    return frame_path


def _build_and_open_backtest_gif(
    frame_paths: list[str],
    backtest_delay_seconds: float,
) -> None:
    """Build an animated GIF from saved backtest frames and open it."""
    if not frame_paths:
        print("No backtest frames were generated.")
        return

    if Image is None:
        print("Pillow is unavailable; opening the final backtest frame only.")
        subprocess.run(["open", frame_paths[-1]], check=False)
        return

    gif_path = os.path.join(os.getcwd(), "backtest_replay.gif")
    images = [Image.open(frame_path) for frame_path in frame_paths]
    frame_duration_ms = max(int(backtest_delay_seconds * 1000), 200)

    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=frame_duration_ms,
        loop=0,
    )
    for image in images:
        image.close()

    subprocess.run(["open", gif_path], check=False)
    print(f"Backtest replay opened: {gif_path}")


def _resolve_backtest_start() -> datetime:
    """Resolve backtest start timestamp from env or recent session fallback."""
    configured_start = os.getenv("BACKTEST_START")
    if configured_start:
        return datetime.fromisoformat(configured_start)

    recent_sessions = _recent_session_timestamps(days_back=10)
    if not recent_sessions:
        raise ValueError("Unable to resolve a recent trading session for backtest")

    return recent_sessions[-1].replace(
        hour=10,
        minute=0,
        second=0,
        microsecond=0,
    )


def _run_snapshot_mode(manager: BotManager, bot_id: str) -> None:
    """Render one polled snapshot and open the saved chart image."""
    polled = _poll_with_fallback(manager, bot_id)

    print(
        "Polled payload:",
        f"candles={len(polled.candles)}",
        f"indicators={len(polled.indicators)}",
    )

    chart = RealtimeTradingChart(
        title="SBIN 5m - Latest Polled Snapshot",
        max_candles=120,
    )
    chart.render(polled)
    _save_and_open_chart(chart, "latest_polled_chart.png")
    _launch_gradio_chart(
        title="SBIN 5m - Latest Polled Snapshot",
        payload=polled,
    )


def _run_backtest_mode(manager: BotManager, bot_id: str) -> None:
    """Advance one bot through historical candles in backtest mode."""
    backtest_steps = int(os.getenv("BACKTEST_STEPS", "100"))
    backtest_delay_seconds = float(os.getenv("BACKTEST_DELAY_SECONDS", "1.0"))
    start_date = _resolve_backtest_start()
    frames_dir = _prepare_backtest_frames_dir()
    frame_paths: list[str] = []

    chart = RealtimeTradingChart(
        title="SBIN 5m - Backtest Replay",
        max_candles=120,
    )
    latest_payload = None

    for step_index in range(backtest_steps):
        latest_payload = manager.poll_bot(
            bot_id=bot_id,
            limit=None,
            emit_updates=False,
            date=start_date if step_index == 0 else None,
            backtest=True,
            backtest_delay_seconds=backtest_delay_seconds,
        )
        chart.render(latest_payload)
        frame_path = _save_backtest_frame(chart, frames_dir, step_index + 1)
        if frame_path is not None:
            frame_paths.append(frame_path)

        last_candle = latest_payload.last_candle
        print(
            f"Backtest step {step_index + 1}/{backtest_steps}:",
            f"candles={len(latest_payload.candles)}",
            f"last_timestamp={getattr(last_candle, 'timestamp', None)}",
            f"last_close={getattr(last_candle, 'close', None)}",
        )

    if latest_payload is not None:
        print(
            "Backtest final payload:",
            f"candles={len(latest_payload.candles)}",
            f"indicators={len(latest_payload.indicators)}",
        )

    _save_and_open_chart(chart, "backtest_chart.png")
    _build_and_open_backtest_gif(frame_paths, backtest_delay_seconds)
    if latest_payload is not None:
        _launch_gradio_chart(
            title="SBIN 5m - Backtest Replay",
            payload=latest_payload,
        )


def main() -> None:
    """Create manager and run either snapshot or backtest chart flow."""
    load_dotenv()
    manager = _build_manager()
    bot_id = "SBIN_5m"

    if os.getenv("BACKTEST_MODE", "true").lower() == "true":
        _run_backtest_mode(manager, bot_id)
        return

    _run_snapshot_mode(manager, bot_id)


if __name__ == "__main__":
    main()
