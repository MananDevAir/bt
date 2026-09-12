#!/bin/bash
# Startup script for 24/7 free hosting (Kerit Cloud, Wispbyte, Pterodactyl panels)
# Set DISCORD_WEBHOOK_URL as an environment variable in your host's panel.

set -e  # Exit immediately if any command fails

echo "========================================="
echo "  Multi-Asset Signal Bot — Starting Up"
echo "========================================="

# Install/upgrade dependencies silently
echo "[1/3] Installing dependencies..."
pip install -r requirements.txt -q

# Show which Python we're using
echo "[2/3] Python: $(python --version)"

# Start the bot in continuous infinite-loop mode with live alerts enabled
echo "[3/3] Launching bot (continuous + live)..."
python run_bot.py --continuous --live
