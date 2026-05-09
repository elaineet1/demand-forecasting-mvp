#!/bin/bash
# Startup script for Railway deployment.
# Launches the Telegram bot as a background process, then starts Streamlit
# in the foreground (Railway monitors the foreground process for health checks).

# Start Telegram bot in background if token is configured
if [ -n "$TELEGRAM_BOT_TOKEN" ]; then
    python telegram_bot.py &
    echo "Telegram bot started (PID $!)"
else
    echo "TELEGRAM_BOT_TOKEN not set — skipping Telegram bot"
fi

# Start Streamlit on Railway's dynamically assigned port
exec python -m streamlit run app.py \
    --server.port "${PORT:-8501}" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false
