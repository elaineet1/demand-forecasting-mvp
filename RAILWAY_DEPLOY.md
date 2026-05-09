# Railway Deployment Guide

Both the Streamlit app and the Telegram bot run in a single Railway service, sharing the same `artifacts/` volume. This mirrors the local setup exactly.

---

## Prerequisites

- Railway account at [railway.app](https://railway.app) (Hobby plan ~$5/month)
- This repo pushed to GitHub
- Your `TELEGRAM_BOT_TOKEN` and `OPENAI_API_KEY` ready

---

## Step 1 — Create a Railway project

1. Go to [railway.app/new](https://railway.app/new)
2. Choose **Deploy from GitHub repo**
3. Select this repository (`demand-forecasting-mvp` or equivalent)
4. Railway will detect Python and start building automatically

---

## Step 2 — Set environment variables

In your Railway project → **Variables** tab, add:

| Variable | Value |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Your bot token from @BotFather |
| `OPENAI_API_KEY` | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-4.1-mini` (or leave unset to use default) |
| `STREAMLIT_APP_URL` | Set this after Step 4 (your Railway public URL) |

> Railway automatically injects `PORT` — do not set it manually.

---

## Step 3 — Add a persistent volume for artifacts

This keeps forecast data alive across redeploys.

1. In your Railway project → **Volumes** tab → **Add Volume**
2. Set **Mount Path** to `/app/artifacts`
3. Size: `1 GB` is more than enough

Without this volume, the bot will lose its data every time Railway redeploys.

---

## Step 4 — Deploy

Railway builds and deploys automatically on every push to your main branch. To trigger manually:

- **Settings** → **Deploy** → **Deploy Now**

Watch the build logs. A successful deploy will show:
```
Telegram bot started (PID ...)
You can now view your Streamlit app in your browser.
```

---

## Step 5 — Get your public URL and update STREAMLIT_APP_URL

1. In Railway → **Settings** → **Networking** → **Generate Domain**
2. Copy the URL (e.g. `https://your-app.railway.app`)
3. Go back to **Variables** and set `STREAMLIT_APP_URL` to that URL
4. Redeploy (Railway will pick up the new variable automatically)

The Telegram bot's `/dashboard` button uses this URL to link back to the app.

---

## How it works

- `start.sh` runs on container start
- Telegram bot starts in the **background** (`python telegram_bot.py &`)
- Streamlit runs in the **foreground** on `$PORT` — Railway uses this for health checks
- Both processes share `/app/artifacts` from the persistent volume
- If Streamlit crashes, Railway restarts the whole container (bot restarts too)

---

## Notes

- **Telegram bot crash recovery:** If the bot crashes independently (not Streamlit), Railway won't auto-restart it since it monitors the foreground process. A simple workaround is to redeploy. For production-grade bot resilience, consider adding `supervisord`.
- **Streamlit Cloud:** You can keep the Streamlit Cloud deployment as a public read-only demo. Just note that data uploaded there won't be shared with the Railway bot.
- **Secrets vs env vars:** On Railway, all secrets go in Variables — no `.streamlit/secrets.toml` needed. The app reads `OPENAI_API_KEY` from `os.environ` first.
