# ML-Assisted Demand & OTB Forecasting MVP

**Version**: 0.3.0  
**Purpose**: Hybrid ML + business-rules forecasting for retail inventory planning  
**Target Users**: Management, planners, and buyers

---

## Live Deployments

| | URL |
|---|---|
| **Web App (Railway)** | https://demand-forecasting-mvp-production.up.railway.app |
| **Web App (Streamlit Cloud — demo)** | https://demand-forecasting-mvp.streamlit.app |
| **Telegram Bot** | Running permanently on Railway alongside the web app |

---

## What It Does

Turns raw Excel/CSV inventory and sales files into a complete monthly planning package:

- **SKU-level demand forecast** — next 1 to 3 months, ML + fallback hierarchy
- **OTB recommendations** — reorder quantity per SKU based on projected stock cover
- **Stock health assessment** — understock / healthy / overstock per item
- **AI narratives** — executive, buyer, and risk summaries via OpenAI
- **RAG chatbot** — ask questions grounded in your actual forecast data
- **Telegram bot** — on-the-go insights and Q&A without opening the web app

---

## Architecture

```
forecasting_mvp/
├── app.py                                    # Entry point — home page, session init, disk restore
├── telegram_bot.py                           # Telegram supervisor bot (RAG chat + commands)
├── start.sh                                  # Railway startup script (bot + Streamlit)
├── railway.toml                              # Railway deployment config
├── requirements.txt
├── pages/
│   ├── 1_Upload_and_Validation.py            # File upload, column mapping, pipeline trigger
│   ├── 2_Executive_Dashboard.py              # KPIs, top actions, stock exposure
│   ├── 3_Forecast_Explorer.py                # SKU-level forecast drill-down
│   ├── 4_OTB_Planner.py                      # Reorder recommendation table
│   ├── 5_Model_Insights.py                   # Feature importance, validation metrics
│   ├── 6_Insights_and_Report_Generator.py    # AI narrative co-pilot
│   └── 7_Forecast_Chat.py                    # Full-page RAG chatbot
├── src/
│   ├── config.py                             # All constants and thresholds
│   ├── state.py                              # Session state init and clear
│   ├── persistence.py                        # Disk save/load for artifacts/
│   ├── io_utils.py                           # File reading and date parsing
│   ├── column_mapper.py                      # Fuzzy column name normalization
│   ├── validators.py                         # Data quality checks
│   ├── preprocess.py                         # Aggregation and master dataset
│   ├── feature_engineering.py               # Lag, rolling mean, trend, calendar features
│   ├── forecasting.py                        # End-to-end pipeline orchestrator
│   ├── fallback.py                           # Fallback forecast hierarchy + M2/M3 estimation
│   ├── model_train.py                        # LightGBM / HistGBM / RandomForest training
│   ├── planner.py                            # OTB calculations and stock health
│   ├── metrics.py                            # WAPE, MAE, RMSE
│   ├── explainability.py                     # Feature importance and fallback stats
│   ├── copilot.py                            # Narrative generation via OpenAI
│   ├── rag.py                                # RAG chat — embeddings, retrieval, sidebar UI
│   └── charts.py                             # Plotly chart helpers
├── artifacts/                                # Runtime artifacts (gitignored)
│   ├── model.pkl                             # Trained ML model
│   ├── feature_cols.pkl                      # Feature column list
│   ├── planner_output.parquet                # Latest planner table
│   ├── planning_summary.json                 # Summary stats
│   ├── model_meta.json                       # Model performance metadata
│   └── run_metadata.json                     # Run timestamp and stats
├── sample_data/
│   ├── vendor_brand_mapping.csv              # Vendor → brand normalization
│   └── event_calendar_template.csv           # Event calendar template
└── data/
    └── simulated/                            # 12-month simulated demo dataset
```

---

## Pipeline

Each upload triggers a 7-step pipeline (`src/forecasting.py`):

1. **Normalize** — fuzzy column mapping, brand normalization
2. **Validate** — check required columns per file type
3. **Merge & aggregate** — SKU-month sales, latest inventory snapshot
4. **Master dataset** — merge_asof joins inventory + event calendar to sales months
5. **Feature engineering** — lag (1, 2m), rolling mean (2m), trend, calendar flags
6. **Train & forecast** — ML model + fallback hierarchy per SKU
7. **OTB planner** — reorder qty, stock health, stock cover months → auto-saved to `artifacts/`

### Forecast hierarchy (per SKU)

| Priority | Method | When used |
|---|---|---|
| 1 | ML model (LightGBM) | ≥ 2 months history, > 20 units/month |
| 2 | Recent 3-month average | Low volume (≤ 20 units/month) |
| 3 | Category + vendor average | No ML, no recent data |
| 4 | Category average | No vendor data |
| 5 | Existing forecast_qty | From inventory file |
| 6 | Zero | No data at all |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web UI | Streamlit |
| Data | Pandas, NumPy |
| ML | LightGBM, Scikit-learn |
| Visualisation | Plotly |
| AI / RAG | OpenAI (`text-embedding-3-small`, `gpt-4.1-mini`) |
| Persistence | joblib (model), Parquet (planner), JSON (summaries) |
| Telegram | python-telegram-bot |
| Hosting | Railway (web + bot), Streamlit Cloud (demo) |

---

## Local Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy secrets template and add your keys
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Add to `.streamlit/secrets.toml`:
```toml
OPENAI_API_KEY = "your_openai_api_key"
OPENAI_MODEL = "gpt-4.1-mini"
```

Add to `.env` (for Telegram bot):
```
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
OPENAI_API_KEY=your_openai_api_key
STREAMLIT_APP_URL=http://localhost:8501
```

Run the app:
```bash
# Streamlit app
python -m streamlit run app.py

# Telegram bot (separate terminal)
python telegram_bot.py
```

---

## Railway Deployment

Both the Streamlit app and the Telegram bot run in a single Railway service sharing a persistent volume at `/app/artifacts`.

See [RAILWAY_DEPLOY.md](RAILWAY_DEPLOY.md) for the full step-by-step setup guide.

**Required Railway environment variables:**

| Variable | Description |
|---|---|
| `TELEGRAM_BOT_TOKEN` | From @BotFather |
| `OPENAI_API_KEY` | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4.1-mini` (or leave unset) |
| `STREAMLIT_APP_URL` | Your Railway public URL |

---

## Telegram Bot Commands

| Command | What it does |
|---|---|
| `/start` | Welcome message and command list |
| `/dashboard` | Link to the web app |
| `/summary` | Portfolio snapshot (SKUs, reorder qty, health counts) |
| `/reorders` | Top 5 SKUs needing reorder |
| `/health` | Understock / healthy / overstock breakdown |
| `/sku ITEM_NO` | Full detail for a specific SKU |
| `/clearchat` | Reset your conversation history |
| Free text | RAG chat — questions answered from live forecast data |

The bot reads from `artifacts/` on disk. Run a forecast in the web app first to populate it.

---

## Data Format

### Inventory files
- Format: `.xls`, `.xlsx`, `.csv`
- Date in filename: e.g. `Item List_18.12.2021.xls`
- Required columns: Item No, Description, Total Stock, Item Status, Active (Y/N)
- Optional: Vendor, Category, Base Price, RRP, Forecast Qty

### Sales files
- Format: `.xls`, `.xlsx`, `.csv`
- Date in filename: e.g. `Sales Analysis_1.9.2021 - 30.9.2021.xls`
- Required columns: Item No, Description, Quantity

### Event calendar (optional)
- Format: `.csv`
- Columns: `date, event_name, event_type, children_day, christmas, school_holiday, year_end_holiday, summer_holiday, campaign_flag, launch_flag`
- Optional scope columns: `vendor`, `manufacturer`
- Template: `sample_data/event_calendar_template.csv`

### Column name recognition

The app automatically recognises common column name variations:

| Standard | Recognised variants |
|---|---|
| `item_no` | Item No, SKU No, Article No |
| `item_description` | Item Desc, Description, Product Description |
| `total_stock` | Total Stock, On Hand, Qty on Hand, Inventory |
| `quantity` | Qty, Qty Sold, Units Sold, Quantity Sold |
| `forecast_qty` | Forecast Qty, Forecast Quantity, Projected Qty |

---

## Key Thresholds (`src/config.py`)

```python
STOCK_COVER_MONTHS_HEALTHY_MIN = 2.0   # Below this = understock risk
STOCK_COVER_MONTHS_HEALTHY_MAX = 3.0   # Above this = overstock risk
REQUEST_LEAD_TIME_MONTHS = 3            # Lead time for purchase planning
MIN_HISTORY_FOR_ML = 2                  # Min monthly observations to use ML
LOW_VOLUME_THRESHOLD = 20               # Units/month below which ML is bypassed
MODEL_OUTLIER_IQR_MULTIPLIER = 3.0      # Outlier clipping threshold
```

---

## Troubleshooting

**Sidebar chatbot does not appear**  
Add `OPENAI_API_KEY` to `.streamlit/secrets.toml` (local) or Railway Variables (deployed).

**Telegram bot has no data to answer from**  
Run a forecast in the web app first — the bot reads from `artifacts/` which is populated after each pipeline run.

**Dates not parsing from filenames**  
Use format `DD.MM.YYYY` or `DD.MM.YYYY - DD.MM.YYYY` in the filename.

**Column mapping warnings**  
Review warnings on the Upload page. Add new variants to `src/column_mapper.py` if needed.

**Insufficient data for ML**  
Upload at least 3 months of files. 6–12 months recommended, 12+ best for seasonality.

**Stock cover shows N/A**  
Expected for new SKUs with no sales history. Populates once sales data exists.

---

## Limitations

- Short history reduces seasonal learning
- Sparse SKUs rely on fallback rules
- Streamlit Cloud has ephemeral storage — artifacts are lost on redeploy (use Railway for persistent use)
- Best used as decision support, not as sole source of truth
- WAPE typically 30–50% for sparse retail data

---

**Last Updated**: May 2026 (v0.3.0 — Railway deployment, Telegram bot, disk persistence)  
**Python**: 3.11+  
**Status**: MVP — production-ready for small-scale use
