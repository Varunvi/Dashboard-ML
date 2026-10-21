"""
=============================================================================
CRUDE OIL PRICE PREDICTION — FINAL REPRODUCIBLE VERSION
Models: SVR & Random Forest with full diagnostic analysis
Seed locked: results identical on every run
FIXES: axis overlaps, 70-30 split, residual % y-axis, RF R² recovery,
       RandomizedSearchCV to prevent stalling
=============================================================================
"""

import os
os.environ['PYTHONHASHSEED'] = '42'

import random
random.seed(42)

import numpy as np
np.random.seed(42)

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# ── Plot Style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 120,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.family': 'sans-serif',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
})
COLORS = {'actual': '#2c3e50', 'svr': '#e74c3c', 'rf': '#27ae60',
          'train': '#3498db', 'test': '#e67e22', 'naive': '#9b59b6'}


# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("=" * 60)
print("1. LOADING DATA")
print("=" * 60)

df = pd.read_excel("Oil prices 2.xlsx")
df.columns = df.columns.str.strip()
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df = df.sort_values('Date').reset_index(drop=True)

print(f"Shape      : {df.shape}")
print(f"Date range : {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nData types:\n{df.dtypes}")


# =============================================================================
# 2. PREPROCESSING
# =============================================================================
print("\n" + "=" * 60)
print("2. PREPROCESSING")
print("=" * 60)

for col in ['Demand (Mbbl/day)', 'USD Index', 'Price (USD/bbl)']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df[['Demand (Mbbl/day)', 'USD Index']] = (
    df[['Demand (Mbbl/day)', 'USD Index']]
    .interpolate(method='linear')
    .bfill().ffill()
)

df['OPEC Policy'] = df['OPEC Policy'].fillna('No Change')
df['OPEC Policy'] = df['OPEC Policy'].map({'Increase': 1, 'No Change': 0, 'Cut': -1})
df = df.dropna(subset=['Price (USD/bbl)'])


# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================
df['Month']           = df['Date'].dt.month
df['Year']            = df['Date'].dt.year
df['Quarter']         = df['Date'].dt.quarter

df['Price_Lag1']      = df['Price (USD/bbl)'].shift(1)
df['Price_Lag2']      = df['Price (USD/bbl)'].shift(2)
df['Price_Lag3']      = df['Price (USD/bbl)'].shift(3)
df['Price_Lag6']      = df['Price (USD/bbl)'].shift(6)

df['Rolling_Mean_3']  = df['Price (USD/bbl)'].rolling(window=3).mean()
df['Rolling_Mean_5']  = df['Price (USD/bbl)'].rolling(window=5).mean()
df['Rolling_Mean_12'] = df['Price (USD/bbl)'].rolling(window=12).mean()
df['Rolling_Std_5']   = df['Price (USD/bbl)'].rolling(window=5).std()
df['Rolling_Std_12']  = df['Price (USD/bbl)'].rolling(window=12).std()

df['Price_MoM_Change']  = df['Price (USD/bbl)'].diff(1)
df['Price_3M_Change']   = df['Price (USD/bbl)'].diff(3)
df['Price_Pct_Change1'] = df['Price (USD/bbl)'].pct_change(1)

df['Demand_x_OPEC'] = df['Demand (Mbbl/day)'] * df['OPEC Policy']
df['USD_x_Demand']  = df['USD Index'] * df['Demand (Mbbl/day)']

df = df.dropna().reset_index(drop=True)

Q1, Q3 = df['Price (USD/bbl)'].quantile([0.25, 0.75])
IQR = Q3 - Q1
outlier_mask = (
    (df['Price (USD/bbl)'] < Q1 - 1.5 * IQR) |
    (df['Price (USD/bbl)'] > Q3 + 1.5 * IQR)
)
print(f"Final shape    : {df.shape}")
print(f"Price outliers : {outlier_mask.sum()} (retained — real market events)")


# =============================================================================
# 4. DATA VISUALIZATION
# =============================================================================
print("\n" + "=" * 60)
print("4. DATA VISUALIZATION")
print("=" * 60)

# Fig 1: Time Series
fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(df['Date'], df['Price (USD/bbl)'],
        color=COLORS['actual'], lw=1.5, label='Oil Price')
if outlier_mask.any():
    ax.scatter(df.loc[outlier_mask, 'Date'],
               df.loc[outlier_mask, 'Price (USD/bbl)'],
               color='red', zorder=5, s=30, label='Outliers (IQR)')
ax.set(title='Crude Oil Price — Time Series',
       xlabel='Date', ylabel='Price (USD/bbl)')
ax.tick_params(axis='x', rotation=30, labelsize=9)
ax.xaxis.labelpad = 10
ax.legend()
plt.tight_layout()
plt.savefig('fig1_timeseries.png', bbox_inches='tight')
plt.show()

# Fig 2: Distribution + Boxplot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df['Price (USD/bbl)'], bins=30,
             color=COLORS['train'], edgecolor='white')
axes[0].set(title='Price Distribution',
            xlabel='Price (USD/bbl)', ylabel='Frequency')
axes[0].tick_params(axis='x', rotation=20)
axes[1].boxplot(df['Price (USD/bbl)'], vert=True, patch_artist=True,
                boxprops=dict(facecolor=COLORS['train'], alpha=0.6))
axes[1].set(title='Price Boxplot', ylabel='Price (USD/bbl)')
plt.tight_layout(pad=2.0)
plt.savefig('fig2_distribution.png', bbox_inches='tight')
plt.show()

# Fig 3: Scatter plots
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, col, cmap in zip(axes,
                          ['Demand (Mbbl/day)', 'USD Index'],
                          ['viridis', 'plasma']):
    sc = ax.scatter(df[col], df['Price (USD/bbl)'],
                    c=df['Year'], cmap=cmap, alpha=0.7, s=20)
    plt.colorbar(sc, ax=ax, label='Year')
    z = np.polyfit(df[col], df['Price (USD/bbl)'], 1)
    ax.plot(np.sort(df[col]),
            np.poly1d(z)(np.sort(df[col])),
            'r--', lw=1.5, label='Trend')
    ax.set(title=f'{col} vs Price', xlabel=col, ylabel='Price (USD/bbl)')
    ax.tick_params(axis='x', rotation=20)
    ax.legend()
plt.tight_layout(pad=2.0)
plt.savefig('fig3_scatter.png', bbox_inches='tight')
plt.show()

# Fig 4: Correlation Heatmap
fig, ax = plt.subplots(figsize=(16, 12))          # bigger canvas
corr = df.drop(columns=['Date']).corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, ax=ax, linewidths=0.5, square=True,
            annot_kws={'size': 7})
ax.set_title('Correlation Heatmap', pad=25, fontsize=13, fontweight='bold')

# X-axis — rotate fully vertical and right-align so nothing overlaps
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=90,          # fully vertical
    ha='right',
    fontsize=7.5
)
# Y-axis — horizontal, slightly larger
ax.set_yticklabels(
    ax.get_yticklabels(),
    rotation=0,
    ha='right',
    fontsize=7.5
)

plt.tight_layout(rect=[0, 0.05, 1, 1])   # bottom margin so x labels aren't cut off
plt.savefig('fig4_heatmap.png', bbox_inches='tight')
plt.show()

# Fig 5: OPEC Policy Boxplot
fig, ax = plt.subplots(figsize=(7, 4))
groups = [df.loc[df['OPEC Policy'] == k, 'Price (USD/bbl)'].values
          for k in [-1, 0, 1]]
ax.boxplot(groups, labels=['Cut', 'No Change', 'Increase'],
           patch_artist=True,
           boxprops=dict(facecolor='#3498db', alpha=0.5))
ax.set(title='Oil Price by OPEC Policy',
       xlabel='OPEC Policy', ylabel='Price (USD/bbl)')
ax.xaxis.labelpad = 8
plt.tight_layout()
plt.savefig('fig5_opec_boxplot.png', bbox_inches='tight')
plt.show()


# =============================================================================
# 5. TRAIN / TEST SPLIT — 70-30
# =============================================================================
FEATURES = [
    'Demand (Mbbl/day)', 'USD Index', 'OPEC Policy',
    'Month', 'Year', 'Quarter',
    'Price_Lag1', 'Price_Lag2', 'Price_Lag3', 'Price_Lag6',
    'Rolling_Mean_3', 'Rolling_Mean_5', 'Rolling_Mean_12',
    'Rolling_Std_5', 'Rolling_Std_12',
    'Price_MoM_Change', 'Price_3M_Change', 'Price_Pct_Change1',
    'Demand_x_OPEC', 'USD_x_Demand',
]

X = df[FEATURES]
y = df['Price (USD/bbl)']

split_idx   = int(0.70 * len(df))
X_train     = X.iloc[:split_idx];  X_test  = X.iloc[split_idx:]
y_train     = y.iloc[:split_idx];  y_test  = y.iloc[split_idx:]
dates_train = df['Date'].iloc[:split_idx]
dates_test  = df['Date'].iloc[split_idx:]

print(f"\nTrain : {len(X_train)} records ({len(X_train)/len(df)*100:.0f}%)")
print(f"Test  : {len(X_test)}  records ({len(X_test)/len(df)*100:.0f}%)")
print(f"Features used : {len(FEATURES)}")

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_sc = scaler_X.fit_transform(X_train)
X_test_sc  = scaler_X.transform(X_test)
y_train_sc = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_sc  = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()


# =============================================================================
# 6. NAIVE BASELINE
# =============================================================================
print("\n" + "=" * 60)
print("6. NAIVE BASELINE  [Price(t) = Price(t-1)]")
print("=" * 60)

naive_pred = df.loc[X_test.index, 'Price_Lag1'].values
naive_mae  = mean_absolute_error(y_test, naive_pred)
naive_rmse = np.sqrt(mean_squared_error(y_test, naive_pred))
naive_r2   = r2_score(y_test, naive_pred)

print(f"  Naive MAE  : {naive_mae:.4f}")
print(f"  Naive RMSE : {naive_rmse:.4f}")
print(f"  Naive R²   : {naive_r2:.4f}")
print("  → Any model must beat this to be considered useful")


# =============================================================================
# 7. BASELINE ML MODELS
# =============================================================================
print("\n" + "=" * 60)
print("7. BASELINE ML MODELS")
print("=" * 60)

svr_base = SVR(kernel='rbf')
svr_base.fit(X_train_sc, y_train_sc)
svr_base_pred = scaler_y.inverse_transform(
    svr_base.predict(X_test_sc).reshape(-1, 1)).ravel()

rf_base = RandomForestRegressor(random_state=42, n_estimators=100)
rf_base.fit(X_train, y_train)
rf_base_pred = rf_base.predict(X_test)

print(f"  SVR Baseline R² : {r2_score(y_test, svr_base_pred):.4f}")
print(f"  RF  Baseline R² : {r2_score(y_test, rf_base_pred):.4f}")


# =============================================================================
# 8. HYPERPARAMETER TUNING — RandomizedSearchCV (fast, no stalling)
# =============================================================================
print("\n" + "=" * 60)
print("8. HYPERPARAMETER TUNING  (RandomizedSearchCV, n_iter=20)")
print("=" * 60)

tscv = TimeSeriesSplit(n_splits=5)

# SVR
param_dist_svr = {
    'kernel' : ['rbf', 'linear'],
    'C'      : [1, 10, 50, 100],
    'gamma'  : ['scale', 0.01, 0.001],
    'epsilon': [0.1, 0.5, 1.0],
}
grid_svr = RandomizedSearchCV(
    SVR(), param_dist_svr,
    n_iter=20,
    cv=tscv,
    scoring='neg_mean_squared_error',
    random_state=42, n_jobs=1, verbose=1
)
grid_svr.fit(X_train_sc, y_train_sc)
best_svr = grid_svr.best_estimator_
print(f"\n  Best SVR params : {grid_svr.best_params_}")

# RF
param_dist_rf = {
    'n_estimators'     : [100, 200, 300],
    'max_depth'        : [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf' : [1, 2, 4],
    'max_features'     : ['sqrt', 'log2'],
}
grid_rf = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_dist_rf,
    n_iter=20,
    cv=tscv,
    scoring='neg_mean_squared_error',
    random_state=42, n_jobs=1, verbose=1
)
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_
print(f"  Best RF  params : {grid_rf.best_params_}")


# =============================================================================
# 9. PREDICTIONS
# =============================================================================
svr_train_pred = scaler_y.inverse_transform(
    best_svr.predict(X_train_sc).reshape(-1, 1)).ravel()
svr_test_pred  = scaler_y.inverse_transform(
    best_svr.predict(X_test_sc).reshape(-1, 1)).ravel()

rf_train_pred = best_rf.predict(X_train)
rf_test_pred  = best_rf.predict(X_test)



from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Select 3 trees
trees = best_rf.estimators_[:3]

for i, tree in enumerate(trees):
    plt.figure(figsize=(12,6))
    plot_tree(tree,
              feature_names=FEATURES,
              filled=True,
              max_depth=3)   # limit depth for clarity
    plt.title(f"Random Forest - Tree {i+1}")
    plt.show()


plt.figure(figsize=(8,5))

plt.scatter(X_train_sc[:, 0], y_train_sc, color='gray', alpha=0.5, label='Data')

plt.scatter(
    X_train_sc[best_svr.support_][:, 0],
    y_train_sc[best_svr.support_],
    color='red', label='Support Vectors'
)

plt.title("SVR Support Vectors")
plt.legend()
plt.show()



# =============================================================================
# 10. EVALUATION
# =============================================================================
print("\n" + "=" * 60)
print("10. MODEL EVALUATION")
print("=" * 60)


def evaluate(name, y_tr, y_tr_p, y_te, y_te_p):
    tr_r2   = r2_score(y_tr, y_tr_p)
    te_mae  = mean_absolute_error(y_te, y_te_p)
    te_rmse = np.sqrt(mean_squared_error(y_te, y_te_p))
    te_r2   = r2_score(y_te, y_te_p)
    mape    = np.mean(np.abs(
        (np.array(y_te) - np.array(y_te_p)) / np.array(y_te))) * 100
    print(f"\n  {'─'*38}")
    print(f"  {name}")
    print(f"  {'─'*38}")
    print(f"  {'Metric':<14} {'Train':>8} {'Test':>8}")
    print(f"  {'R²':<14} {tr_r2:>8.4f} {te_r2:>8.4f}")
    print(f"  {'MAE':<14} {'—':>8} {te_mae:>8.4f}")
    print(f"  {'RMSE':<14} {'—':>8} {te_rmse:>8.4f}")
    print(f"  {'MAPE (%)':<14} {'—':>8} {mape:>8.2f}")
    return dict(tr_r2=tr_r2, te_mae=te_mae,
                te_rmse=te_rmse, te_r2=te_r2, mape=mape)


svr_res = evaluate("SVR (Tuned)", y_train, svr_train_pred, y_test, svr_test_pred)
rf_res  = evaluate("RF  (Tuned)", y_train, rf_train_pred,  y_test, rf_test_pred)

print(f"\n  {'─'*38}")
print("  NAIVE BASELINE")
print(f"  {'─'*38}")
print(f"  R²   : {naive_r2:.4f}")
print(f"  MAE  : {naive_mae:.4f}")
print(f"  RMSE : {naive_rmse:.4f}")
print(f"\n  {'─'*38}")
print("  IMPROVEMENT OVER NAIVE BASELINE")
print(f"  {'─'*38}")
print(f"  SVR R² gain : {svr_res['te_r2'] - naive_r2:+.4f}")
print(f"  RF  R² gain : {rf_res['te_r2']  - naive_r2:+.4f}")


# =============================================================================
# 11. LEARNING CURVES
# =============================================================================
print("\n" + "=" * 60)
print("11. LEARNING CURVES")
print("=" * 60)


def plot_learning_curve(estimator, X, y, title, color, ax, is_scaled=False):
    train_sizes = np.linspace(0.2, 1.0, 8)
    train_sz, train_sc, val_sc = [], [], []

    for size in train_sizes:
        n = max(int(size * len(X)), 10)
        X_tr = X[:n]
        y_tr = y[:n]

        if is_scaled:
            sX = StandardScaler()
            sy = StandardScaler()
            X_tr_s = sX.fit_transform(X_tr)
            y_tr_s = sy.fit_transform(y_tr.values.reshape(-1, 1)).ravel()
            X_va_s = sX.transform(X[n:]) if n < len(X) else X_tr_s
            y_va   = y[n:] if n < len(X) else y_tr
            estimator.fit(X_tr_s, y_tr_s)
            tr_pred = sy.inverse_transform(
                estimator.predict(X_tr_s).reshape(-1, 1)).ravel()
            va_pred = sy.inverse_transform(
                estimator.predict(X_va_s).reshape(-1, 1)).ravel()
            y_va_vals = y_va.values if hasattr(y_va, 'values') else y_va
        else:
            estimator.fit(X_tr, y_tr)
            tr_pred   = estimator.predict(X_tr)
            va_pred   = estimator.predict(X[n:]) if n < len(X) else tr_pred
            y_va_vals = y[n:].values if n < len(X) else y_tr.values

        train_sz.append(n)
        train_sc.append(r2_score(y_tr, tr_pred))
        val_sc.append(r2_score(y_va_vals, va_pred) if len(y_va_vals) > 1 else np.nan)

    ax.plot(train_sz, train_sc, 'o-', color=color, label='Train R²', lw=1.5)
    ax.plot(train_sz, val_sc,   's--', color=color, alpha=0.6,
            label='Validation R²', lw=1.5)
    ax.axhline(0, color='gray', lw=0.8, linestyle=':')
    ax.set(title=title, xlabel='Training Samples', ylabel='R²')
    ax.tick_params(axis='x', rotation=20)
    ax.legend(fontsize=8)
    ax.set_ylim(-0.5, 1.05)


fig, axes = plt.subplots(1, 2, figsize=(13, 5))
plot_learning_curve(SVR(**grid_svr.best_params_),
                    X_train_sc, y_train,
                    'SVR — Learning Curve', COLORS['svr'], axes[0],
                    is_scaled=True)
plot_learning_curve(RandomForestRegressor(**grid_rf.best_params_,
                                          random_state=42),
                    X_train, y_train,
                    'RF — Learning Curve', COLORS['rf'], axes[1])
plt.suptitle('Learning Curves — Generalization Behavior',
             fontsize=13, fontweight='bold')
plt.tight_layout(pad=2.5)
plt.savefig('fig_learning_curves.png', bbox_inches='tight')
plt.show()
print("  → Converging train/val curves = good generalization")
print("  → Large persistent gap        = overfitting")


# =============================================================================
# 12. FEATURE IMPORTANCE
# =============================================================================
print("\n" + "=" * 60)
print("12. FEATURE IMPORTANCE")
print("=" * 60)

perm = permutation_importance(
    best_svr, X_test_sc, y_test_sc,
    n_repeats=20, random_state=42, scoring='r2'
)
svr_imp_df = (
    pd.DataFrame({
        'Feature'   : FEATURES,
        'Importance': perm.importances_mean,
        'Std'       : perm.importances_std
    }).sort_values('Importance', ascending=True)
)
print(svr_imp_df.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(15, 7))
axes[0].barh(svr_imp_df['Feature'], svr_imp_df['Importance'],
             xerr=svr_imp_df['Std'],
             color=COLORS['svr'], edgecolor='white', alpha=0.85)
axes[0].axvline(0, color='black', lw=0.8, linestyle='--')
axes[0].set(title='SVR — Permutation Feature Importance\n'
                   '(drop in R² when feature is shuffled)',
            xlabel='Mean decrease in R²')
axes[0].tick_params(axis='y', labelsize=8)

rf_imp_df = (
    pd.DataFrame({
        'Feature'   : FEATURES,
        'Importance': best_rf.feature_importances_
    }).sort_values('Importance', ascending=True)
)
axes[1].barh(rf_imp_df['Feature'], rf_imp_df['Importance'],
             color=COLORS['rf'], edgecolor='white', alpha=0.85)
axes[1].set(title='RF — Built-in Feature Importance\n'
                   '(mean decrease in impurity)',
            xlabel='Importance Score')
axes[1].tick_params(axis='y', labelsize=8)

plt.suptitle('Feature Importance: SVR vs RF', fontsize=13, fontweight='bold')
plt.tight_layout(pad=2.5)
plt.savefig('fig_feature_importance.png', bbox_inches='tight')
plt.show()


# =============================================================================
# 13. RESULT VISUALIZATIONS
# =============================================================================

# Actual vs Predicted
fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=False)
for ax, name, pred, color in zip(
        axes, ['SVR', 'Random Forest'],
        [svr_test_pred, rf_test_pred],
        [COLORS['svr'], COLORS['rf']]):
    ax.plot(dates_test, y_test.values,
            color=COLORS['actual'], lw=1.8, label='Actual', zorder=3)
    ax.plot(dates_test, pred, color=color, lw=1.4,
            linestyle='--', label=f'{name} Predicted', zorder=2)
    ax.plot(dates_test, naive_pred, color=COLORS['naive'],
            lw=1, linestyle=':', label='Naive Baseline', zorder=1)
    r2  = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    ax.set(title=f'{name} — Actual vs Predicted  |  R²={r2:.4f}  MAE={mae:.2f}',
           ylabel='Price (USD/bbl)')
    ax.legend(loc='upper left', fontsize=8)
    ax.tick_params(axis='x', rotation=30, labelsize=9)
    ax.xaxis.labelpad = 10
plt.tight_layout(pad=2.5)
plt.savefig('fig_actual_vs_predicted.png', bbox_inches='tight')
plt.show()

# Residual Plots
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
for row, (name, pred) in enumerate([('SVR', svr_test_pred),
                                      ('Random Forest', rf_test_pred)]):
    resid = y_test.values - pred
    color = COLORS['svr'] if 'SVR' in name else COLORS['rf']
    n_obs = len(resid)

    axes[row, 0].plot(dates_test, resid, color=color, lw=1)
    axes[row, 0].axhline(0, color='black', lw=1, linestyle='--')
    axes[row, 0].set(title=f'{name} — Residuals over Time',
                     xlabel='Date', ylabel='Residual (USD/bbl)')
    axes[row, 0].tick_params(axis='x', rotation=30, labelsize=9)
    axes[row, 0].xaxis.labelpad = 8

    weights = np.ones(n_obs) / n_obs * 100
    axes[row, 1].hist(resid, bins=25, weights=weights,
                      color=color, edgecolor='white', alpha=0.8)
    axes[row, 1].axvline(0, color='black', lw=1.5, linestyle='--')
    axes[row, 1].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f%%'))
    axes[row, 1].set(title=f'{name} — Residual Distribution',
                     xlabel='Residual (USD/bbl)',
                     ylabel='Percentage of Observations (%)')
    axes[row, 1].tick_params(axis='x', rotation=20, labelsize=9)
    axes[row, 1].xaxis.labelpad = 8

plt.suptitle('Residual Analysis', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout(pad=2.5)
plt.savefig('fig_residuals.png', bbox_inches='tight')
plt.show()

# Model Comparison Bar Chart
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
metrics  = ['R²',             'MAE',            'RMSE']
svr_vals = [svr_res['te_r2'], svr_res['te_mae'], svr_res['te_rmse']]
rf_vals  = [rf_res['te_r2'],  rf_res['te_mae'],  rf_res['te_rmse']]
nv_vals  = [naive_r2,         naive_mae,          naive_rmse]

for ax, metric, sv, rfv, nv in zip(axes, metrics, svr_vals, rf_vals, nv_vals):
    x    = np.arange(3)
    bars = ax.bar(x, [sv, rfv, nv],
                  color=[COLORS['svr'], COLORS['rf'], COLORS['naive']],
                  edgecolor='white', width=0.5)
    for bar, val in zip(bars, [sv, rfv, nv]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005 * max(abs(sv), abs(rfv), abs(nv)),
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set(title=f'Test {metric}', ylabel=metric,
           xticks=x, xticklabels=['SVR', 'RF', 'Naive'])
    ax.tick_params(axis='x', labelsize=9)

plt.suptitle('Model Comparison — SVR vs RF vs Naive Baseline',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout(pad=2.0)
plt.savefig('fig_model_comparison.png', bbox_inches='tight')
plt.show()

# Overfitting Check
fig, ax = plt.subplots(figsize=(8, 5))
labels = ['SVR Train', 'SVR Test', 'RF Train', 'RF Test']
scores = [svr_res['tr_r2'], svr_res['te_r2'],
          rf_res['tr_r2'],  rf_res['te_r2']]
colors = [COLORS['svr'], COLORS['svr'], COLORS['rf'], COLORS['rf']]
alphas = [0.5, 1.0, 0.5, 1.0]

for i, (label, score, c, a) in enumerate(zip(labels, scores, colors, alphas)):
    ax.bar(i, score, color=c, alpha=a, edgecolor='white', width=0.6)
    ax.text(i, score + 0.005, f'{score:.3f}', ha='center', fontsize=10)

ax.set(xticks=range(4),
       title='Overfitting Check — Train vs Test R²',
       ylabel='R²')
ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
ax.axhline(1.0, color='gray', lw=0.8, linestyle='--')
plt.tight_layout()
plt.savefig('fig_overfitting.png', bbox_inches='tight')
plt.show()


# =============================================================================
# 14. SAVE RESULTS
# =============================================================================
df.loc[X_test.index, 'SVR_Prediction']   = svr_test_pred
df.loc[X_test.index, 'RF_Prediction']    = rf_test_pred
df.loc[X_test.index, 'Naive_Prediction'] = naive_pred
df.loc[X_test.index, 'SVR_Residual']     = y_test.values - svr_test_pred
df.loc[X_test.index, 'RF_Residual']      = y_test.values - rf_test_pred

df.to_excel("final_predictions_improved.xlsx", index=False)
print("\nSaved: final_predictions_improved.xlsx")
print("Saved: all fig_*.png figures")