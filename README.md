# ML-Assisted Demand & OTB Forecasting MVP

**Version**: 0.1.0 (MVP)  
**Purpose**: Hybrid ML + business-rules forecasting for inventory planning  
**Target Users**: Management and purchasing teams

---

## Table of Contents

- [Quick Start](#quick-start)
- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Technical Stack](#technical-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Data Format Requirements](#data-format-requirements)
- [Model Details](#model-details)
- [Limitations & Assumptions](#limitations--assumptions)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites

- Python 3.11+
- pip or conda

### Setup (5 minutes)

```bash
# Navigate to project folder
cd forecasting_mvp

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## Project Overview

### What It Does

This MVP combines machine learning with fallback business rules to forecast sales and recommend purchase quantities for Company's toy distribution business:

1. **Data Ingestion**: Accepts inventory and sales Excel/CSV files
2. **Data Validation**: Normalizes inconsistent column names, checks quality
3. **Feature Engineering**: Creates lag, rolling, temporal, and categorical features
4. **ML Training**: Trains gradient boosting model on historical sales
5. **Forecasting**: Predicts next 3 months of demand per SKU
6. **Fallback Logic**: Uses category/vendor averages for sparse SKUs
7. **OTB Planning**: Recommends purchase quantities based on stock targets

### Key Features

✅ **Time-Aware ML Validation** - Respects time-series structure  
✅ **Hybrid ML + Fallback** - Graceful degradation for sparse data  
✅ **Stock Health Assessment** - Categorizes risk (understock/healthy/overstock)  
✅ **Stock Cover Months** - Current stock ÷ latest monthly sales per SKU  
✅ **Explainability** - Feature importance and per-SKU explanations  
✅ **Downloadable Output** - CSV/Excel/JSON planner table  
✅ **Multi-page Dashboard** - Executive, detail, and insight views  
✅ **Ask Your Data (RAG Chatbot)** - Free-form Q&A on forecast data and uploaded documents

---

## Architecture

```
forecasting_mvp/
├── app.py                          # Main Streamlit app
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── .streamlit/
│   └── config.toml                 # Streamlit configuration
├── pages/
│   ├── 1_Upload_and_Validation.py          # File intake and validation
│   ├── 2_Executive_Dashboard.py            # High-level metrics
│   ├── 3_Forecast_Explorer.py              # SKU-level drill-down
│   ├── 4_OTB_Planner.py                    # Downloadable recommendations (OTB)
│   ├── 5_Model_Insights.py                 # Performance and transparency
│   ├── 6_Insights_and_Report_Generator.py  # AI-generated narratives and reports
│   └── 7_Forecast_Chat.py                  # Full-page RAG chatbot
├── src/
│   ├── __init__.py
│   ├── config.py                   # Global constants and thresholds
│   ├── state.py                    # Session state management
│   ├── io_utils.py                 # File I/O and date parsing
│   ├── column_mapper.py            # Column name normalization
│   ├── validators.py               # Data quality checks
│   ├── preprocess.py               # Cleaning and aggregation
│   ├── feature_engineering.py      # Feature creation
│   ├── model_train.py              # ML model training
│   ├── forecasting.py              # End-to-end pipeline
│   ├── fallback.py                 # Fallback forecasting rules
│   ├── planner.py                  # OTB planning logic
│   ├── metrics.py                  # Evaluation metrics
│   ├── explainability.py           # Transparency and explanations
│   ├── rag.py                      # RAG chatbot (embeddings, retrieval, sidebar UI)
│   └── charts.py                   # Plotly visualizations
├── sample_data/                    # (Empty; for samples/templates)
└── outputs/                        # (Empty; for downloaded files)
```

---

## Technical Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Streamlit | Lightweight web UI |
| **Data** | Pandas, NumPy | Data manipulation |
| **ML** | LightGBM, Scikit-learn | Gradient boosting regression |
| **Viz** | Plotly | Interactive charts |
| **AI / RAG** | OpenAI (`text-embedding-3-small`, `gpt-4.1-mini`) | Embeddings and chat completions |
| **Document Parsing** | pypdf | PDF text extraction for RAG |
| **Utilities** | Openpyxl, xlrd, Python-dateutil | File I/O and dates |

### Model Selection

- **Primary**: LightGBM (fast, interpretable)
- **Fallback 1**: HistGradientBoostingRegressor (scikit-learn)
- **Fallback 2**: RandomForestRegressor (scikit-learn)

---

## Installation

### Standard Environment (Recommended)

```bash
# Clone/download project
cd forecasting_mvp

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install from requirements.txt
pip install -r requirements.txt
```

### Conda Environment (Alternative)

```bash
conda create -n company_forecast python=3.11 -y
conda activate company_forecast
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import streamlit, pandas, lightgbm; print('✓ All imports OK')"
```

---

## Usage

### Step 1: Prepare Your Data

**Inventory Files**:
- Excel or CSV format
- Include date in filename (e.g., `Item List_18.12.2021.xls`)
- Required columns: Item No, Description, Total Stock, Item Status, Active (Y/N)
- Optional: Vendor, Category, Base Price, RRP, Forecast Qty

**Sales Files**:
- Excel or CSV format
- Include date or date range in filename (e.g., `Sales Analysis_1.9.2021 - 30.9.2021.xls`)
- Required columns: Item No, Description, Quantity
- Optional: Customer Name, Sales Amount, Gross Profit

**Event Calendar (Optional)**:
- CSV with columns: date, event_name, event_type, children_day, christmas, school_holiday, year_end_holiday, summer_holiday, campaign_flag, launch_flag
- See `sample_data/event_calendar_template.csv` for template
- Optional scope columns: vendor (brand), category, manufacturer

**Brand Mapping (Optional)**:
- CSV with columns: vendor_raw, brand
- If provided, vendor values are normalized to brand names before forecasting
- See `sample_data/vendor_brand_mapping.csv` for template

### Step 2: Run the App

```bash
streamlit run app.py
```

Open browser to `http://localhost:8501`

### Optional: Enable OpenAI for AI Features

OpenAI powers two features: the **Insights & Report Generator** page (business narrative summaries) and the **Ask Your Data chatbot** (sidebar Q&A on every page).

1. Install dependencies (already included in `requirements.txt`):

```bash
pip install -r requirements.txt
```

2. Add your OpenAI key to `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "your_openai_api_key"
OPENAI_MODEL = "gpt-4.1-mini"
```

You can start from the included template file:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

3. Run the app — the sidebar chatbot will appear on every page, and the **Insights & Report Generator** and **Forecast Chat** pages will be fully enabled.

> **Streamlit Community Cloud**: Instead of a local `secrets.toml`, go to your app's **Settings → Secrets** in the Streamlit Cloud dashboard and paste the key-value pairs there.

### Step 3: Navigate Pages

1. **Upload & Validation**: Upload files, review column mapping
2. **Executive Dashboard**: Review high-level metrics and health distribution
3. **Forecast Explorer**: Drill into individual SKUs, sales history, and stock cover months
4. **OTB Planner**: Review and download recommendation table with scenario planning
5. **Model Insights**: Check model performance and assumptions
6. **Insights & Report Generator**: AI-generated narrative summaries and business reports
7. **Forecast Chat**: Full-page conversational Q&A on your forecast data

### Ask Your Data (RAG Chatbot)

A sidebar chatbot is available on every page once an OpenAI key is configured. It uses Retrieval-Augmented Generation (RAG) to answer questions grounded in your actual forecast data.

**What you can ask:**
- "Which SKUs are understocked this month?"
- "What is the reorder quantity for brand X?"
- "Which items have the highest overstock risk?"
- "Summarise the stock health across all categories."

**Uploading documents:**  
Expand the **Upload a document** section in the sidebar to attach PDF or TXT files (e.g. supplier price lists, buying policies). The chatbot will include those documents as additional context when answering questions.

**How it works:**
1. Forecast data is automatically converted into text documents and embedded using `text-embedding-3-small`.
2. Embeddings are cached in session state — no re-embedding on every message.
3. The most relevant chunks (forecast data + any uploaded files) are retrieved by cosine similarity and passed to `gpt-4.1-mini` as context.

---

## Data Format Requirements

### Column Name Recognition

The app automatically recognizes common column name variations:

| Standard Name | Recognized Variants |
|---------------|-------------------|
| `item_no` | Item No, Item No., SKU No, Article No |
| `item_description` | Item Desc, Description, Product Description |
| `total_stock` | Total Stock, On Hand, Qty on Hand, Inventory |
| `warehouse_stock` | Warehouse Stock, WH Stock, $warehseStock |
| `sku_number` | SKU Number, SKU No, SKUNUMBER |
| `quantity` | Qty, Qty Sold, Units Sold, Quantity Sold |
| `sales_amt` | Sales Amount, Sales Value, Revenue |
| `forecast_qty` | Forecast Qty, Forecast Quantity, Projected Qty |

### Date Parsing

Filenames can include dates in these formats:
- `Item_List_18.12.2021.xls` → 2021-12-18
- `Sales_1.9.2021_-_30.9.2021.xls` → 2021-09-30 (end date)
- `Inventory_2021-12-18.csv` → 2021-12-18

---

## Model Details

### Training Approach

- **Target**: Monthly quantity sold (SKU-level aggregation)
- **Validation**: Time-Series Split (no random shuffle)
- **Train/Test Ratio**: 80/20 chronological split
- **Data Readiness**:
  - Minimum to run: 3 months
  - Recommended: 6–12 months
  - Best for seasonality: 12+ months

### Features Created

**Temporal**:
- Year, month, quarter, day-of-week, week-of-year

**Lag Features**:
- Previous 1-month and 2-month sales

**Rolling Statistics**:
- 2-month rolling mean and standard deviation

**Categorical (One-Hot Encoded)**:
- Vendor, manufacturer, category, subcategory, item_status

**Price/Stock**:
- Margin (RRP - Base Price)
- Stock cover (months of supply)
- Warehouse stock ratio

**Event Calendar (If Provided)**:
- Children's day flag
- Christmas flag
- School holidays flag
- Year-end holidays flag
- Summer holidays flag
- Campaign flag
- Launch flag

### Hyperparameters

**LightGBM**:
```python
{
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1,
    'num_leaves': 31,
}
```

---

## Limitations & Assumptions

### Data Limitations

1. **Limited History**: Seasonal patterns are harder to learn with short or uneven history
2. **Sparse SKUs**: Short or low-volume histories still require fallback rules
3. **New Launches**: Frequent new products lack historical data
4. **Aggregate Only**: Treats each SKU independently (no cannibalization modeling)

### Model Limitations

1. **Accuracy**: WAPE likely 30-50%+ due to sparse data
2. **No Trend Breaking**: Assumes demand stability
3. **No Deep Learning**: Gradient boosting only (for interpretability)
4. **Local Deployment**: MVP not production-hardened

### Business Assumptions

1. **Active Filter**: Only "Active = Y" SKUs in final recommendations
2. **Stock Coverage**: 2-3 months considered healthy (configurable)
3. **Lead Time**: 3 months assumed for purchase planning
4. **Fallback Hierarchy**:
   - ML model (if ≥2 months data)
   - Category + Vendor average
   - Category average
   - Existing forecast_qty from inventory file
   - Zero (no data)

---

## Key Thresholds (Configurable in `src/config.py`)

```python
STOCK_COVER_MONTHS_HEALTHY_MIN = 2.0      # Understock risk threshold
STOCK_COVER_MONTHS_HEALTHY_MAX = 3.0      # Overstock risk threshold
REQUEST_LEAD_TIME_MONTHS = 3               # Lead time for purchase planning
MIN_HISTORY_FOR_ML = 2                     # Min monthly observations for ML
MIN_SALES_MONTHS_FOR_SEASONAL = 4          # (Future: seasonal modeling)
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'lightgbm'"

**Solution**: LightGBM is optional; the app falls back to HistGradientBoosting
```bash
pip install lightgbm
```

### Issue: "No sales files uploaded" error

**Cause**: CSV files may not have been recognized  
**Solution**: Ensure files are `.csv`, `.xls`, or `.xlsx`

### Issue: "Column mapping warnings for many columns"

**Cause**: File has non-standard column names  
**Solution**: Review warnings; add new variants to `src/column_mapper.py` if needed

### Issue: "Insufficient featured data" warning

**Cause**: Not enough SKU-month combinations for ML  
**Solution**: Collect more data or adjust `MIN_HISTORY_FOR_ML` in config.py

### Issue: Dates not parsing from filenames

**Cause**: Filename doesn't include recognizable date  
**Solution**: Use format like `Item_List_DD.MM.YYYY.xls` or `Sales_DD.MM.YYYY - DD.MM.YYYY.xls`

### Issue: Memory error with large files

**Cause**: Too many rows to process  
**Solution**: Pre-filter files to recent months; split into smaller uploads

### Issue: Sidebar chatbot does not appear

**Cause**: OpenAI API key not configured  
**Solution**: Add `OPENAI_API_KEY` to `.streamlit/secrets.toml` (local) or Streamlit Cloud Secrets (deployed)

### Issue: "OpenAI package required" error on Forecast Chat page

**Solution**:
```bash
pip install openai pypdf
```
Both are included in `requirements.txt` — ensure you have the latest dependencies installed.

### Issue: Stock Cover (months) shows as N/A

**Cause**: No sales data available for that SKU (`latest_monthly_sales` and `recent_3m_avg` are both zero)  
**Solution**: This is expected for new/inactive SKUs with no sales history. The column will populate once sales data exists.

---

## Performance & Deployment Notes

### Local Performance

- **Typical Runtime**: 10-30 seconds for upload to forecast
- **Max File Size**: 200 MB per file (configurable in config.py)
- **Memory Usage**: ~1 GB for typical dataset

### Streamlit Community Cloud (Recommended for Team Sharing)

1. Push the repository to GitHub (the repo must be public, or you must have a Streamlit Cloud account linked to a private repo).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repo, branch (`main`), and entry file (`app.py`).
4. Click **Advanced settings → Secrets** and paste:

```toml
OPENAI_API_KEY = "your_openai_api_key"
OPENAI_MODEL = "gpt-4.1-mini"
```

5. Click **Deploy**. The app will be live at a public URL within a few minutes.

> **Do not commit `.streamlit/secrets.toml`** — it is already in `.gitignore`. Manage all secrets through the Streamlit Cloud UI.

### Self-Hosted / Docker

```bash
# Option 1: Self-hosted
streamlit run app.py --server.port 8501 --server.address 0.0.0.0

# Option 2: Docker
docker build -t company_forecast .
docker run -p 8501:8501 company_forecast
```

---

## Contributing / Customization

### Adding New Features

1. **Custom Fallback Logic**: Edit `src/fallback.py`
2. **New Features**: Add to `src/feature_engineering.py`
3. **New Visualizations**: Add to `src/charts.py`
4. **New Business Rules**: Edit `src/planner.py`

### Modifying Thresholds

Edit `src/config.py`:

```python
STOCK_COVER_MONTHS_HEALTHY_MIN = 1.5  # More conservative
STOCK_COVER_MONTHS_HEALTHY_MAX = 4.0  # More generous
```

### Adding Column Variants

Edit `src/column_mapper.py` `COLUMN_VARIANTS` dict:

```python
'my_new_standard_name': [
    'variant 1',
    'variant 2',
]
```

---

## Next Steps / Roadmap

### Potential V1.1 Enhancements

- [ ] Prophet integration for seasonal modeling
- [ ] SHAP values for model explainability
- [ ] Multi-SKU bundling logic
- [ ] Demand elasticity estimation
- [ ] Markdown management (clearance pricing)
- [ ] Supplier lead-time variation
- [ ] SQLite persistence layer
- [ ] API for external integrations

---

## Support & Contact

- **Data Issues**: Check column mapping in Upload & Validation page
- **Model Accuracy**: Review assumptions in Model Insights page
- **Technical Issues**: Install all requirements; verify Python version
- **Feature Requests**: Document in project notes

---

## License & Attribution

**ML-Assisted Demand & OTB Forecasting MVP v0.1**  
Date: 2026  

This MVP is provided as-is for educational and planning support purposes.

---

**Last Updated**: April 2026 (v0.2.0 — RAG chatbot, Insights & Report Generator, stock cover months)  
**Python Version**: 3.11  
**Status**: MVP (Production-ready for small-scale use)
