"""CLI utilities for tradingbot package initialization and setup."""

import subprocess
import sys


def init_command():
    """Initialize tradingbot - install Playwright browsers and setup."""
    print("\n" + "=" * 60)
    print("Initializing tradingbot package...")
    print("=" * 60)

    try:
        print("\n📦 Installing Playwright browsers...")
        subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            check=True,
        )
        print("✓ Playwright browsers installed successfully!")

        print("\n" + "=" * 60)
        print("✓ Tradingbot initialization complete!")
        print("=" * 60)
        print("\nYou can now use tradingbot. Copy .env.example to .env and update")
        print("your credentials before running the application.")

    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error during initialization (exit code {e.returncode})")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    init_command()
