import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.tree import plot_tree
from engine import engine, FEATURES

if not engine.is_trained:
    engine.train_models()

df = engine.df
results = engine.results
meta = results['meta']
metrics = results['metrics']
svr_res = metrics['svr']
rf_res = metrics['rf']
naive_metrics = metrics['naive']
preds = results['test_predictions']

COLORS = {'actual': '#2c3e50', 'svr': '#e74c3c', 'rf': '#27ae60',
          'train': '#3498db', 'test': '#e67e22', 'naive': '#9b59b6'}

# 1. Timeseries
fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(df['Date'], df['Price (USD/bbl)'], color=COLORS['actual'], lw=1.5, label='Oil Price')
if df['Is_Outlier'].any():
    ax.scatter(df.loc[df['Is_Outlier'], 'Date'], df.loc[df['Is_Outlier'], 'Price (USD/bbl)'],
               color='red', zorder=5, s=30, label='Outliers (IQR)')
ax.set(title='Crude Oil Price — Time Series', xlabel='Date', ylabel='Price (USD/bbl)')
ax.legend()
plt.tight_layout()
plt.savefig('fig1_timeseries.png', bbox_inches='tight')
plt.close()

# 2. Distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df['Price (USD/bbl)'], bins=30, color=COLORS['train'], edgecolor='white')
axes[0].set(title='Price Distribution', xlabel='Price (USD/bbl)', ylabel='Frequency')
axes[1].boxplot(df['Price (USD/bbl)'], vert=True, patch_artist=True,
                boxprops=dict(facecolor=COLORS['train'], alpha=0.6))
axes[1].set(title='Price Boxplot', ylabel='Price (USD/bbl)')
plt.tight_layout(pad=2.0)
plt.savefig('fig2_distribution.png', bbox_inches='tight')
plt.close()

# 3. Scatters
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, col, cmap in zip(axes, ['Demand (Mbbl/day)', 'USD Index'], ['viridis', 'plasma']):
    sc = ax.scatter(df[col], df['Price (USD/bbl)'], c=df['Year'], cmap=cmap, alpha=0.7, s=20)
    plt.colorbar(sc, ax=ax, label='Year')
    z = np.polyfit(df[col], df['Price (USD/bbl)'], 1)
    ax.plot(np.sort(df[col]), np.poly1d(z)(np.sort(df[col])), 'r--', lw=1.5, label='Trend')
    ax.set(title=f'{col} vs Price', xlabel=col, ylabel='Price (USD/bbl)')
    ax.legend()
plt.tight_layout(pad=2.0)
plt.savefig('fig3_scatter.png', bbox_inches='tight')
plt.close()

# 4. Correlation Heatmap
fig, ax = plt.subplots(figsize=(16, 12))
corr = df.drop(columns=['Date', 'Is_Outlier'], errors='ignore').corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn', center=0, ax=ax, linewidths=0.5, square=True, annot_kws={'size': 7})
ax.set_title('Correlation Heatmap', pad=25, fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('fig4_heatmap.png', bbox_inches='tight')
plt.close()

# 5. OPEC Boxplot
fig, ax = plt.subplots(figsize=(7, 4))
groups = [df.loc[df['OPEC Policy'] == k, 'Price (USD/bbl)'].values for k in [-1, 0, 1]]
ax.boxplot(groups, labels=['Cut', 'No Change', 'Increase'], patch_artist=True, boxprops=dict(facecolor='#3498db', alpha=0.5))
ax.set(title='Oil Price by OPEC Policy', xlabel='OPEC Policy', ylabel='Price (USD/bbl)')
plt.tight_layout()
plt.savefig('fig5_opec_boxplot.png', bbox_inches='tight')
plt.close()

# 6. Actual vs Predicted
dates_test = pd.to_datetime(preds['dates'])
y_test = np.array(preds['actual'])
fig, axes = plt.subplots(2, 1, figsize=(13, 9))
for ax, name, p_vals, color in zip(axes, ['SVR', 'Random Forest'], [preds['svr'], preds['rf']], [COLORS['svr'], COLORS['rf']]):
    ax.plot(dates_test, y_test, color=COLORS['actual'], lw=1.8, label='Actual')
    ax.plot(dates_test, p_vals, color=color, lw=1.4, linestyle='--', label=f'{name} Predicted')
    ax.plot(dates_test, preds['naive'], color=COLORS['naive'], lw=1, linestyle=':', label='Naive Baseline')
    ax.set(title=f'{name} — Actual vs Predicted', ylabel='Price (USD/bbl)')
    ax.legend()
plt.tight_layout(pad=2.5)
plt.savefig('fig_actual_vs_predicted.png', bbox_inches='tight')
plt.close()

# 7. Model Comparison
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
metrics_list = ['R²', 'MAE', 'RMSE']
svr_vals = [svr_res['te_r2'], svr_res['te_mae'], svr_res['te_rmse']]
rf_vals  = [rf_res['te_r2'],  rf_res['te_mae'],  rf_res['te_rmse']]
nv_vals  = [naive_metrics['te_r2'], naive_metrics['te_mae'], naive_metrics['te_rmse']]

for ax, metric, sv, rfv, nv in zip(axes, metrics_list, svr_vals, rf_vals, nv_vals):
    x = np.arange(3)
    bars = ax.bar(x, [sv, rfv, nv], color=[COLORS['svr'], COLORS['rf'], COLORS['naive']], edgecolor='white', width=0.5)
    for bar, val in zip(bars, [sv, rfv, nv]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set(title=f'Test {metric}', ylabel=metric, xticks=x, xticklabels=['SVR', 'RF', 'Naive'])
plt.suptitle('Model Comparison — SVR vs RF vs Naive Baseline', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout(pad=2.0)
plt.savefig('fig_model_comparison.png', bbox_inches='tight')
plt.close()

# 8. Overfitting Check
fig, ax = plt.subplots(figsize=(8, 5))
labels = ['SVR Train', 'SVR Test', 'RF Train', 'RF Test']
scores = [svr_res['tr_r2'], svr_res['te_r2'], rf_res['tr_r2'], rf_res['te_r2']]
colors = [COLORS['svr'], COLORS['svr'], COLORS['rf'], COLORS['rf']]
alphas = [0.5, 1.0, 0.5, 1.0]
for i, (label, score, c, a) in enumerate(zip(labels, scores, colors, alphas)):
    ax.bar(i, score, color=c, alpha=a, edgecolor='white', width=0.6)
    ax.text(i, score + 0.005, f'{score:.3f}', ha='center', fontsize=10)
ax.set(xticks=range(4), xticklabels=labels, title='Overfitting Check — Train vs Test R²', ylabel='R²')
plt.tight_layout()
plt.savefig('fig_overfitting.png', bbox_inches='tight')
plt.close()

print("All diagnostic figures generated and saved successfully!")
