# AI Sales Forecasting Dashboard 

---

## Training Plan Requirements Checklist

| Requirement | File | Status |
|---|---|---|
| Data ingestion | `app/data_pipeline.py` → `ingest_data()` | ✅ All 5 Kaggle files merged |
| Data cleaning pipeline | `app/data_pipeline.py` → `clean_data()` | ✅ Dedup, interpolate, fill gaps |
| Sales forecasting model | `app/forecast_model.py` | ✅ Prophet + regressors |
| Model evaluation | `train.py` → `evaluate_model()` | ✅ MAE + MAPE on 30-day holdout |
| Interactive dashboard | `app/static/dashboard.html` | ✅ Chart.js, live slider |
| FastAPI backend | `app/main.py` | ✅ 5 endpoints + logging |
| Docker deployment | `Dockerfile` | ✅ |
| GitHub workflow | This README → Git section | ✅ Feature branching |
| Documentation | README.md | ✅ |

---

## About the Dataset

**Kaggle: Store Sales — Time Series Forecasting**
(Corporación Favorita, Ecuadorian grocery chain)
https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data

### All 5 files and how they are used

| File | Columns | How it's used in this project |
|---|---|---|
| `train.csv` | date, store_nbr, family, sales, onpromotion | Main source — aggregated to daily total sales + daily total promotions |
| `stores.csv` | store_nbr, city, state, type, cluster | Provides store metadata (available for future feature engineering) |
| `oil.csv` | date, dcoilwtico | Daily oil price merged as a regressor — Ecuador's economy is oil-dependent, so oil price correlates with consumer spending |
| `holidays_events.csv` | date, type, locale, locale_name, description, transferred | National holidays flagged as `is_holiday=1` and passed as a regressor — holidays significantly change sales patterns |
| `transactions.csv` | date, store_nbr, transactions | Daily total transactions merged as a feature showing market activity volume |

**Why use all 5 files instead of just `train.csv`?**
`train.csv` alone gives you raw sales numbers. The other files give
context — *why* sales spiked on a particular day (holiday?), *what*
economic conditions existed (oil price), and *how busy* stores were
(transactions). Prophet uses these as extra "regressors" which improves
forecast accuracy compared to using sales history alone.

### What the pipeline does with them
1. Loads all 5 files
2. Aggregates `train.csv` from 3 million rows to ~1,700 daily rows
3. Merges oil prices, holiday flags, and transaction counts per day
4. Cleans: removes duplicates, fills gaps, interpolates missing values
5. Outputs one clean DataFrame: `date, sales, onpromotion, oil_price, is_holiday, transactions`

---

## Project Structure

```
sales_forecast/
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
├── requirements.txt
├── train.py                      # run this first
├── app/
│   ├── __init__.py
│   ├── data_pipeline.py          # ingestion + cleaning (all 5 files)
│   ├── forecast_model.py         # Prophet model + evaluation + recommendation
│   ├── main.py                   # FastAPI backend (5 endpoints + logging)
│   └── static/
│       └── dashboard.html        # interactive dashboard (Chart.js)
├── data/                         # place all 5 Kaggle CSV files here
│   ├── train.csv
│   ├── stores.csv
│   ├── oil.csv
│   ├── holidays_events.csv
│   └── transactions.csv
└── models/                       # auto-created after training
    └── forecast_model.pkl
```

---

## Step-by-Step: How to Run

### Step 1 — Download the dataset

1. Go to: https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data
2. Sign in (free account). Click **"Join Competition"** and accept terms.
3. Click **"Download All"** → unzip the downloaded file.
4. Copy these 5 files into your project's `data/` folder:
   - `train.csv`
   - `stores.csv`
   - `oil.csv`
   - `holidays_events.csv`
   - `transactions.csv`

### Step 2 — Open in VS Code and set up environment

```bash
# In VS Code terminal (inside the project folder)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

Then: `Ctrl+Shift+P` → "Python: Select Interpreter" → pick `.\venv\Scripts\python.exe`
Open a **new** terminal afterward.

### Step 3 — Verify data loaded correctly

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/train.csv')
print('Shape:', df.shape)
print(df.head(3))
"
```
Expected: Shape around (3000888, 6)

### Step 4 — Train the model

```bash
python train.py
```

This runs all 4 steps and prints:
```
Step 1: Data ingestion + cleaning...
Step 2: Evaluating model (30-day holdout)...
  MAE:  ...
  MAPE: ...%
Step 3: Training final model on full dataset...
Step 4: Saving model...
  Model saved -> models/forecast_model.pkl
Training complete.
```

### Step 5 — Run the API + Dashboard

```bash
uvicorn app.main:app --reload
```

Open in browser:
- `http://127.0.0.1:8000` → **Interactive dashboard**
- `http://127.0.0.1:8000/docs` → API documentation
- `http://127.0.0.1:8000/health` → Health check
- `http://127.0.0.1:8000/api/summary` → Dataset statistics
- `http://127.0.0.1:8000/api/historical` → Historical data
- `http://127.0.0.1:8000/api/forecast?days=30` → Forecast + recommendation

### Step 6 — Docker

```bash
# Build the image
docker build -t sales-forecast .

# Run the container
docker run -p 8000:8000 \
  -v $(pwd)/models:/code/models \
  -v $(pwd)/data:/code/data \
  sales-forecast
```

**Why the `-v` flags?** They mount your local `data/` and `models/`
folders into the container — that way the container can read the CSV
files and the trained model without having to bake large data files
into the image itself.

Test same URLs as Step 5.

Push to Docker Hub (optional):
```bash
docker login
docker tag sales-forecast yourusername/sales-forecast
docker push yourusername/sales-forecast
```

---

## Git Workflow — Feature by Feature

You said you only know direct `git push`. Here is the correct workflow
with feature branching, explained simply then demonstrated.

### Core concept

**Never work directly on `main`.** Each feature gets its own branch.
When done, you merge it into `main` via a Pull Request on GitHub.
This gives you a clean history and protects `main` from broken code.

### One-time setup

```bash
git init
git add .
git commit -m "Initial project setup"
# Create a new repo on GitHub.com first, then:
git remote add origin https://github.com/YOURUSERNAME/sales-forecast.git
git branch -M main
git push -u origin main
```

### Feature branch workflow (repeat for each feature)

```bash
# 1. Create and switch to a new branch
git checkout -b feature/data-pipeline

# 2. Do your work (edit files, test them)

# 3. Stage and commit your changes
git add .
git commit -m "Add data ingestion and cleaning pipeline"

# 4. Push the branch to GitHub
git push -u origin feature/data-pipeline

# 5. On GitHub.com: open a Pull Request -> merge it

# 6. Come back to main, pull the merged changes, delete old branch
git checkout main
git pull origin main
git branch -d feature/data-pipeline
```

### Suggested branches for this project (do them in this order)

```
feature/data-pipeline        # app/data_pipeline.py done
feature/forecast-model       # app/forecast_model.py + train.py done
feature/fastapi-backend      # app/main.py done
feature/dashboard-ui         # app/static/dashboard.html done
feature/docker               # Dockerfile + .dockerignore done
```

### Key commands to know

```bash
git status                   # see what changed
git branch                   # list branches (* = current)
git checkout -b <name>       # create + switch to new branch
git checkout main            # switch back to main
git add .                    # stage all changes
git commit -m "message"      # save a snapshot with a description
git push                     # upload to GitHub
git pull                     # download latest from GitHub
git fetch                    # see remote changes WITHOUT merging yet
git log --oneline            # view commit history
```

---

## API Endpoints Summary

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Interactive dashboard |
| GET | `/health` | Model + data status |
| GET | `/api/summary` | Dataset statistics |
| GET | `/api/historical` | Last 180 days of sales |
| GET | `/api/forecast?days=N` | N-day forecast + recommendation |
