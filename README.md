# 🛢️ Petro-Predict AI: Crude Oil Price Forecasting & Diagnostic Dashboard

An end-to-end Machine Learning web application and diagnostic suite for Crude Oil Price Prediction. This project implements Support Vector Regression (**SVR**) and Random Forest Regressor (**RF**) alongside a **Naive Baseline** ($Price_t = Price_{t-1}$), backed by cross-validation, residual diagnostics, exploratory data analysis (EDA), and a real-time **What-If Scenario Simulator**.

Built with **FastAPI**, **Scikit-Learn**, **Tailwind CSS**, and **Chart.js**.

---

## 📸 Key Features

- **🏆 Model Benchmark & Performance Scorecards**:
  - Live side-by-side comparison of **SVR (Tuned)**, **Random Forest**, and **Naive Baseline**.
  - Comprehensive metrics: **$R^2$ Score**, **MAE** (Mean Absolute Error), **RMSE**, and **MAPE (%)**.
  - **Overfitting Diagnostics**: Visual train vs. test score comparison to ensure generalization.

- **📈 Interactive Time-Series Forecasting**:
  - Out-of-sample test predictions overlaid with actual historical settlement prices.
  - Interactive series toggles with hover tooltips and dynamic zooming.

- **📉 Residual & Error Diagnostics**:
  - **Residuals Over Time** ($e_t = y_t - \hat{y}_t$) to assess variance and homoscedasticity.
  - **Residual Percentage Distribution Histogram** for normality and zero-centered error verification.

- **🔬 Exploratory Data Analysis (EDA)**:
  - Full historical timeline with **Tukey's IQR Outlier Detection** ($1.5 \times IQR$).
  - Settlement price frequency distribution histogram with summary statistics.
  - **OPEC Policy Stance Impact**: Historical price distributions under *Production Cut (-1)*, *No Change (0)*, and *Production Increase (+1)*.
  - Scatter plots with trendlines for **Global Demand vs. Price** and **USD Index (DXY) vs. Price**.

- **🧠 Explainability & Interpretability**:
  - **SVR Permutation Feature Importance** (mean drop in test $R^2$ score upon feature shuffling).
  - **Random Forest Gini Impurity Importance** ranking.
  - **Learning Curves**: Sample size progression curves for both estimators to detect bias vs. variance tradeoffs.
  - **Decision Tree Anatomist**: Extraction of primary split conditions, thresholds, and sample distributions from the ensemble.
  - **Support Vector Distribution**: Density and percentage of train set support vectors defining the $\epsilon$-tube.

- **🎛️ Real-Time "What-If" Scenario Simulator**:
  - Interactive sliders for **Demand (Mbbl/day)**, **USD Index (DXY)**, **OPEC Policy (-1, 0, +1)**, and **Previous Month Price ($Lag_1$)**.
  - Instant live prediction recalculation comparing SVR, Random Forest, and Consensus Ensemble.

- **💾 Dataset Explorer & Excel Export**:
  - Searchable, filterable, and paginated data table displaying engineered features, predictions, and residuals.
  - 1-click export to [`final_predictions_improved.xlsx`](final_predictions_improved.xlsx).

- **🔄 Dataset Uploader & Custom Hyperparameter Tuning**:
  - Drag-and-drop file upload for updated `.xlsx` or `.csv` files.
  - Configurable train/test split ratio (e.g. 70/30) and `RandomizedSearchCV` cross-validation iteration count.

---

## 📁 Repository Structure

```
dashboard/
├── app.py                         # FastAPI web server and REST API endpoints
├── engine.py                      # Core ML modeling, feature engineering & tuning pipeline
├── run_dashboard.py               # Launcher script with auto-browser launch
├── start_dashboard.bat            # 1-click Windows batch launcher
├── requirements.txt               # Python package dependencies
├── .gitignore                     # Git ignore rules
├── generate_sample_data.py         # Realistic benchmark dataset generator
├── generate_figures.py            # Static diagnostic figures generator
├── Oil prices 2.xlsx              # Benchmark dataset (or your custom data)
├── final_predictions_improved.xlsx# Generated test predictions and residuals export
├── static/
│   └── app.js                     # Frontend interactive dashboard logic & Chart.js rendering
└── templates/
    └── index.html                 # Responsive dashboard UI (Tailwind CSS + Chart.js)
```

---

## ⚡ Quick Start Guide

### 1. Prerequisites
- **Python 3.10+** installed.
- Check your Python installation:
  ```bash
  python --version
  ```

### 2. Install Dependencies
Clone the repository and install required packages:
```bash
git clone <repository-url>
cd dashboard
pip install -r requirements.txt
```

### 3. Launch the Dashboard

**Option A (1-Click for Windows):**
Double-click `start_dashboard.bat`.

**Option B (Command Line):**
```bash
python run_dashboard.py
```
Or directly with Uvicorn:
```bash
python -m uvicorn app:app --port 8000
```

Open your browser to:
👉 **[http://localhost:8000](http://localhost:8000)**

---

## 📊 Using Your Own Dataset

By default, the dashboard includes a synthesized historical benchmark dataset reflecting real-world oil market dynamics (2012–2024).

To use your own dataset:

### Option 1: Via Dashboard UI (Recommended)
1. Open the dashboard at `http://localhost:8000`.
2. Navigate to the **"Upload & Tuning"** tab.
3. Drag & drop or select your `.xlsx` or `.csv` file.
4. Click **"Upload & Retrain Pipeline"**.

### Option 2: Place File in Root Directory
Replace `Oil prices 2.xlsx` in the project root with your file and click **"Retrain Pipeline Now"** under the Upload & Tuning tab.

### Expected Data Columns
| Column Name | Type | Description | Example Values |
| :--- | :--- | :--- | :--- |
| `Date` | Date / String | Monthly date timestamp | `01-01-2022`, `2022-01-01` |
| `Price (USD/bbl)` | Numeric | Crude oil settlement target price | `75.50`, `82.10` |
| `Demand (Mbbl/day)` | Numeric | Global consumption volumes | `98.5`, `102.1` |
| `USD Index` | Numeric | US Dollar Index (DXY) | `95.2`, `104.0` |
| `OPEC Policy` | String / Integer | Production decision | `Cut` (-1), `No Change` (0), `Increase` (1) |

---

## 🤖 Machine Learning Pipeline Details

### 1. Feature Engineering
From the raw inputs, the pipeline automatically extracts 20 predictive features:
- **Calendar Signals**: `Month`, `Year`, `Quarter`
- **Autoregressive Lags**: `Price_Lag1`, `Price_Lag2`, `Price_Lag3`, `Price_Lag6`
- **Rolling Windows**: 3, 5, and 12-month rolling means; 5 and 12-month rolling standard deviations
- **Momentum Indicators**: Month-over-Month change, 3-Month change, 1-Month percentage change
- **Interaction Terms**: `Demand_x_OPEC` ($Demand \times OPEC\ Policy$), `USD_x_Demand` ($USD \times Demand$)

### 2. Time-Series Validation
To prevent look-ahead bias and temporal leakage, cross-validation strictly uses **`TimeSeriesSplit` (5 folds)**.

### 3. Estimator Configurations
- **Support Vector Regressor (SVR)**:
  - Scaled features via `StandardScaler`.
  - Hyperparameter search space: `kernel` (`rbf`, `linear`), `C` ($1 \dots 100$), `gamma` (`scale`, $0.01$, $0.001$), `epsilon` ($0.1, 0.5, 1.0$).
- **Random Forest Regressor (RF)**:
  - Hyperparameter search space: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`.
- **Naive Baseline**:
  - $Price_t = Price_{t-1}$. Serves as the minimal performance hurdle for any ML model.

---

## 🌐 REST API Endpoints

The dashboard exposes a clean JSON REST API:

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/` | `GET` | Dashboard Web Interface |
| `/api/status` | `GET` | Engine status and record counts |
| `/api/results` | `GET` | Trained model performance metrics, forecasts, residuals & importances |
| `/api/eda` | `GET` | EDA datasets (timeseries with outlier flags, distributions, OPEC stats) |
| `/api/predict` | `POST` | Real-time scenario prediction simulation |
| `/api/retrain` | `POST` | Trigger model retraining with custom split and iteration parameters |
| `/api/upload` | `POST` | Upload custom dataset file (`.xlsx`, `.csv`) and retrain |
| `/api/data` | `GET` | Paginated dataset records for table view |
| `/api/export` | `GET` | Download `final_predictions_improved.xlsx` |

### Example Simulation Request:
```bash
curl -X POST "http://localhost:8000/api/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "demand": 101.5,
    "usd_index": 103.5,
    "opec_policy": -1,
    "lag1": 78.5,
    "month": 10,
    "year": 2024
  }'
```

---

## 📄 License
This project is open-source and available under the MIT License.
