#!/bin/bash
# Setup script for tradingbot - installs package and runs initialization

set -e

echo "========================================"
echo "Setting up tradingbot package..."
echo "========================================"

# Install the package
echo ""
echo "1/2: Installing tradingbot package..."
uv pip install -e .

# Run initialization
echo ""
echo "2/2: Running initialization..."
uv run tradingbot-init

echo ""
echo "========================================"
echo "✓ Setup complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Copy .env.example to .env"
echo "2. Update your credentials in .env"
echo "3. Run: uv run python main.py"
