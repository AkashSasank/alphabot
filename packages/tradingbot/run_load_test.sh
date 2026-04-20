#!/bin/bash

# Load test runner for TradingBot sequence/indicator performance
# Usage: ./run_load_test.sh [options]
#
# Examples:
#   ./run_load_test.sh                    # Standard 5 min test, 50 users, 5 spawn rate
#   ./run_load_test.sh --duration 10m     # Run for 10 minutes
#   ./run_load_test.sh --users 100        # Test with 100 concurrent users

DURATION=${DURATION:-"5m"}
USERS=${USERS:-"50"}
SPAWN_RATE=${SPAWN_RATE:-"5"}
HEADLESS=${HEADLESS:-"--headless"}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --duration)
            DURATION="$2"
            shift 2
            ;;
        --users)
            USERS="$2"
            shift 2
            ;;
        --spawn-rate)
            SPAWN_RATE="$2"
            shift 2
            ;;
        --web)
            HEADLESS=""
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "🚀 Starting TradingBot Load Test"
echo "================================"
echo "Duration:    $DURATION"
echo "Users:       $USERS (spawned at $SPAWN_RATE/sec)"
echo "Mode:        $([ -z '$HEADLESS' ] && echo 'Web UI (http://localhost:8089)' || echo 'Headless')"
echo ""
echo "Test Configuration:"
echo "  • Sequence length: 50-100 candles (random)"
echo "  • Indicators per ticker: 5-20 (random)"
echo "  • Operations: build_sequence, update_candle, add_candle"
echo ""

cd "$(dirname "$0")" || exit 1

# Use parent venv
VENV_PATH="${PWD}/../../.venv/bin"
$VENV_PATH/python -m locust -f locustfile.py $HEADLESS -u "$USERS" -r "$SPAWN_RATE" -t "$DURATION"
