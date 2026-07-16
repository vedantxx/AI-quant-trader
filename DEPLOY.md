# Deploying the dashboard on Streamlit Community Cloud

The repo is deploy-ready. Community Cloud builds straight from GitHub — free,
public URL, auto-redeploys on every push to `main`.

## Prerequisites

- The repo is on GitHub and public: `vedantxx/AI-quant-trader` ✔
- `requirements.txt` lists all deps (incl. `streamlit`) ✔
- Entry point: `ui/streamlit_app.py` ✔
- An Alpaca **paper** key/secret and a password for the dashboard.

## Steps

1. Go to **https://share.streamlit.io** and sign in with GitHub.
2. **Create app** → **Deploy a public app from GitHub**.
3. Fill in:
   - Repository: `vedantxx/AI-quant-trader`
   - Branch: `main`
   - Main file path: `ui/streamlit_app.py`
   - (Advanced) Python version: **3.11** or **3.12**.
4. Open **Advanced settings → Secrets** and paste (values from your paper
   account — see `.streamlit/secrets.toml.example`):
   ```toml
   ALPACA_API_KEY = "your_paper_api_key"
   ALPACA_SECRET_KEY = "your_paper_secret_key"
   APP_PASSWORD = "choose-a-strong-password"
   ```
5. **Deploy**. First build installs deps + compiles hmmlearn (a few minutes).
6. Open the URL, enter `APP_PASSWORD`, click **Start / Retrain**.

Secrets set here are injected as environment variables, so `AlpacaClient`
(reads `ALPACA_API_KEY` / `ALPACA_SECRET_KEY`) works with no `.env` on the host.

## Security — read before hosting

- **The URL is public.** `APP_PASSWORD` is the only gate; choose a strong one.
- **Keep it paper.** Leave `broker.paper_trading: true` in
  [config/settings.yaml](config/settings.yaml). Never put live keys in cloud secrets.
- **Dry-run defaults ON.** Live-order mode still requires unchecking the toggle;
  it will submit to the paper broker only.
- Never commit real keys. `.env` and `.streamlit/secrets.toml` are gitignored.

## Resource notes

- Community Cloud gives ~1 GB RAM. The app defaults to **Fast HMM** (trimmed
  candidates / restarts) so startup is seconds, not minutes. Leave it on.
- Auto-refresh polls the broker each interval; 15–30 s is polite.

## Alternative: self-host with Docker

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "ui/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t ai-quant-trader .
docker run -p 8501:8501 \
  -e ALPACA_API_KEY=... -e ALPACA_SECRET_KEY=... -e APP_PASSWORD=... \
  ai-quant-trader
```
