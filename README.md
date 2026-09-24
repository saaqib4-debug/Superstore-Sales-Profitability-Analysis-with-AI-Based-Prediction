# Superstore Sales & Profitability Analysis with AI-Based Prediction

**IBM SkillsBuild Data Analytics with AI — Academic Internship Project**

**Student:** Mahammadsakib Mulla

---

## Project Overview

An end-to-end data analytics and AI project using the Superstore retail dataset (9,994 transactions, 2014–2017). The project covers exploratory data analysis, business KPI calculations, machine learning for sales prediction and profitability classification, a REST API backend, and an interactive web dashboard.

---

## Problem Statement

A US retail chain faces two critical challenges:
1. **~19.4% of all transactions result in a loss** — the business needs to understand what drives these losses and be able to predict them *before* an order ships.
2. **Sales are unevenly distributed** across product categories and regions, requiring predictive insights to guide sales strategy.

---

## Objectives

- Perform comprehensive Exploratory Data Analysis (EDA)
- Calculate real business KPIs from actual data
- Build a **Sales Prediction** regression model
- Build a **Profitability Classification** model (profitable vs loss-making)
- Deploy models through a FastAPI backend and Streamlit dashboard

---

## Dataset

| Property | Value |
|---|---|
| File | `data/superstore.csv` |
| Source | Sample - Superstore (Tableau / IBM SkillsBuild) |
| Records | 9,994 transactions |
| Columns | 21 |
| Period | January 2014 – December 2017 |
| Geography | United States (49 states, 531 cities) |
| Categories | Furniture, Office Supplies, Technology |

---

## Technologies

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.9+ | Core language |
| pandas | ≥1.5 | Data manipulation |
| numpy | ≥1.23 | Numerical operations |
| matplotlib / seaborn | ≥3.6 / ≥0.12 | Visualizations |
| scikit-learn | ≥1.2 | ML models & evaluation |
| joblib | ≥1.2 | Model serialization |
| FastAPI | ≥0.95 | REST API backend |
| uvicorn | ≥0.20 | ASGI server |
| Streamlit | ≥1.22 | Web dashboard frontend |
| Plotly | ≥5.14 | Interactive charts |

---

## Project Structure

```
Superstore_AI_Project/
│
├── data/
│   └── superstore.csv              ← Dataset
│
├── notebooks/
│   └── Superstore_Complete_Analysis.ipynb  ← Complete analysis notebook
│
├── src/
│   └── analysis_pipeline.py        ← EDA + ML training pipeline (Phases 2–8)
│
├── backend/
│   ├── __init__.py
│   └── main.py                     ← FastAPI REST API
│
├── frontend/
│   └── app.py                      ← Streamlit web dashboard
│
├── models/                         ← Saved ML models (auto-generated)
│   ├── best_sales_model.pkl
│   ├── best_profit_classifier.pkl
│   ├── scaler.pkl
│   ├── feature_columns.json
│   └── encoding_info.json
│
├── outputs/
│   ├── charts/                     ← 23 saved visualizations
│   └── reports/                    ← KPIs, insights, model results (JSON)
│
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone / navigate to project root

```bash
cd IBM/Superstore_AI_Project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **macOS note:** XGBoost requires OpenMP. If XGBoost fails, run `brew install libomp` or the project will automatically skip XGBoost and use Gradient Boosting instead.

---

## How to Run

### Step 1 — Run the Analysis Pipeline (Phases 2–8)

This trains all models, generates charts, and saves all artifacts.

```bash
# From the project root (IBM/Superstore_AI_Project/)
python src/analysis_pipeline.py
```

Expected output: 23 charts in `outputs/charts/`, models in `models/`, JSON reports in `outputs/reports/`.

---

### Step 2 — Run the Jupyter Notebook

```bash
# Open in JupyterLab or VS Code
jupyter notebook notebooks/Superstore_Complete_Analysis.ipynb
```

Or open directly in VS Code using the Jupyter extension.

**Note:** The notebook reads from `../data/superstore.csv` — run it from the `notebooks/` directory, or adjust `DATA_PATH` in Cell 2.

---

### Step 3 — Start the FastAPI Backend

```bash
# From the project root
uvicorn backend.main:app --reload --port 8000
```

The API will be available at: `http://127.0.0.1:8000`  
Interactive docs: `http://127.0.0.1:8000/docs`

---

### Step 4 — Start the Streamlit Frontend

```bash
# From the project root (in a separate terminal)
streamlit run frontend/app.py
```

The dashboard will open at: `http://localhost:8501`

---

## API Endpoints

### GET Endpoints

| Endpoint | Description |
|---|---|
| `GET /` | API overview & available endpoints |
| `GET /health` | Health check and model status |
| `GET /kpis` | All business KPIs |
| `GET /sales-trend` | Monthly sales data |
| `GET /profit-trend` | Monthly profit data |
| `GET /category-performance` | Sales & profit by category |
| `GET /subcategory-performance` | Full sub-category breakdown |
| `GET /region-performance` | Region analytics |
| `GET /segment-performance` | Segment analytics |
| `GET /product-performance` | Top/bottom products & customers |
| `GET /model-performance` | ML model metrics & feature importance |

### POST Endpoints

| Endpoint | Input | Returns |
|---|---|---|
| `POST /predict-sales` | Order features JSON | Predicted sales ($) |
| `POST /predict-profitability` | Order features JSON | Profitable / Loss-Making + probability |

**Sample POST request:**
```json
{
  "category": "Technology",
  "sub_category": "Phones",
  "region": "West",
  "segment": "Consumer",
  "ship_mode": "Standard Class",
  "quantity": 3,
  "discount": 0.2,
  "order_month": 6,
  "order_quarter": 2,
  "order_year": 2017,
  "shipping_delay": 4
}
```

---

## ML Models

### Sales Prediction (Regression)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | $211.53 | $481.10 | 0.2834 |
| Ridge Regression | $211.43 | $481.10 | 0.2834 |
| Random Forest | $198.01 | $521.43 | 0.1582 |
| Gradient Boosting | $198.69 | $575.11 | −0.0240 |

**Best model: Linear Regression (R² = 0.2834)**

> Note: Moderate R² is expected — Sales is heavily influenced by product type variance (Copiers: $20k+ vs Paper: $5–50). The model is most reliable for standard product categories.

### Profitability Classification

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.91 | 0.99 | 0.90 | 0.94 | 0.9817 |
| Random Forest | 0.94 | 0.97 | 0.96 | 0.96 | **0.9819** |
| Gradient Boosting | 0.94 | 0.94 | 0.98 | 0.96 | 0.9805 |

**Best model: Random Forest (ROC-AUC = 0.9819)**

---

## Key Findings

| Finding | Detail |
|---|---|
| Loss rate | 19.4% of transactions are loss-making |
| #1 Loss driver | Discount — correlation with Profit = −0.22 |
| Loss-making sub-categories | Tables (−$17,725), Bookcases (−$3,473), Supplies (−$1,189) |
| Most profitable region | West — 14.9% profit margin |
| Least profitable region | Central — 7.9% profit margin |
| Best profit margin segment | Home Office — 14.0% |
| Top product | Canon imageCLASS 2200 Copier — $25,200 profit |
| Worst products | Cubify 3D Printers — $12,720 combined losses |
| Seasonal peak | Q4 (Oct–Dec) highest sales & profit |

---

## Business Recommendations

1. **Cap discounts at 20%** — discounts ≥ 30% produce average losses of $107/order
2. **Review Tables & Bookcases pricing** — both run at negative margin
3. **Deploy loss-prediction model at order entry** — ROC-AUC = 0.98
4. **Focus growth in West region** — highest profit margin
5. **Target Home Office and Corporate segments** — better margins than Consumer
6. **Prepare for Q4 demand** — plan inventory/staffing for seasonal surge
7. **Push Copier/High-Margin Technology** — best ROI per sales dollar
8. **Discontinue Cubify 3D Printer SKUs** — $12.7k in losses

---

## License

This project is submitted by **Mahammadsakib Mulla** as part of the IBM SkillsBuild Data Analytics with AI Academic Internship.
