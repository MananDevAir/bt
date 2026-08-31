#!/bin/bash
# setup.sh - Installs dependencies and configures the systemd service
set -e

echo "🚀 Setting up Signal Bot..."

# 1. Ensure we are in the bot directory
BOT_DIR=$(pwd)
if [ ! -f "run_bot.py" ]; then
    echo "❌ Error: Please run this script from the root of the bot repository."
    echo "   cd /path/to/bot && bash deploy/setup.sh"
    exit 1
fi

# 2. Check for .env file
if [ ! -f ".env" ]; then
    echo "⚠️ Warning: .env file not found. Copying .env.example..."
    cp .env.example .env
    echo "⚠️ Please edit .env with your Telegram and API keys before starting the bot!"
fi

# 3. Install Python 3.13 and venv if missing (Ubuntu/Debian)
if ! command -v python3.13 &> /dev/null; then
    echo "📦 Installing Python 3.13..."
    sudo apt update
    sudo apt install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt update
    sudo apt install -y python3.13 python3.13-venv
fi

# 4. Create virtual environment
if [ ! -d "venv" ]; then
    echo "🐍 Creating Python virtual environment..."
    python3.13 -m venv venv
fi

# 5. Install dependencies
echo "📚 Installing Python dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 6. Configure systemd service
echo "⚙️ Configuring systemd service..."
SERVICE_FILE="/etc/systemd/system/signalbot.service"

# Create a temporary file for the service with replaced variables
TMP_SERVICE=$(mktemp)
cp deploy/signalbot.service $TMP_SERVICE
sed -i "s|USERNAME|$(whoami)|g" $TMP_SERVICE
sed -i "s|BOT_DIR|$BOT_DIR|g" $TMP_SERVICE

# Install the service
sudo mv $TMP_SERVICE $SERVICE_FILE
sudo chown root:root $SERVICE_FILE
sudo chmod 644 $SERVICE_FILE

# 7. Enable and start the service
echo "🔄 Reloading systemd daemon..."
sudo systemctl daemon-reload
sudo systemctl enable signalbot

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the bot, run:"
echo "  sudo systemctl start signalbot"
echo ""
echo "To view logs in real-time, run:"
echo "  sudo journalctl -u signalbot -f"
