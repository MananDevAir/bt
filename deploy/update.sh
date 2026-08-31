#!/bin/bash
# update.sh - Pulls latest code, updates dependencies, and restarts the bot
set -e

echo "🔄 Updating Signal Bot..."

# 1. Ensure we are in the bot directory
BOT_DIR=$(pwd)
if [ ! -f "run_bot.py" ]; then
    echo "❌ Error: Please run this script from the root of the bot repository."
    echo "   cd /path/to/bot && bash deploy/update.sh"
    exit 1
fi

# 2. Pull latest code from GitHub
echo "📥 Pulling latest code..."
git pull origin main

# 3. Update dependencies if required
echo "📚 Updating Python dependencies..."
source venv/bin/activate
pip install -r requirements.txt

# 4. Restart the service
echo "🚀 Restarting the signalbot service..."
sudo systemctl restart signalbot

echo "✅ Update complete! The bot is running."
echo "To view logs, run: sudo journalctl -u signalbot -f"
