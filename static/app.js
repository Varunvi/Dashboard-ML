// Crude Oil Machine Learning Dashboard Application Logic

let modelResults = null;
let edaData = null;
let currentResidualModel = 'svr';
let currentSimOpec = 0;
let tableCurrentPage = 1;
let searchDebounceTimer = null;

// Chart references
let charts = {};

document.addEventListener('DOMContentLoaded', async () => {
  if (window.lucide) {
    lucide.createIcons();
  }
  setupDragAndDrop();
  await loadDashboardData();
});

// Tab Switching
function switchTab(tabId) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.classList.remove('active', 'text-blue-400', 'bg-blue-500/10');
    btn.classList.add('text-slate-400');
  });

  const targetTab = document.getElementById(tabId);
  if (targetTab) {
    targetTab.classList.remove('hidden');
  }

  const activeBtn = document.querySelector(`[data-tab="${tabId}"]`);
  if (activeBtn) {
    activeBtn.classList.add('active', 'text-blue-400', 'bg-blue-500/10');
    activeBtn.classList.remove('text-slate-400');
  }

  if (window.lucide) lucide.createIcons();

  // Resize charts on tab display
  setTimeout(() => {
    Object.values(charts).forEach(c => {
      if (c && typeof c.resize === 'function') c.resize();
    });
  }, 100);
}

// Fetch Core Dashboard Data
async function loadDashboardData() {
  showLoading(true, 'Loading model predictions and diagnostics...');
  try {
    const [resResp, edaResp] = await Promise.all([
      fetch('/api/results'),
      fetch('/api/eda')
    ]);

    modelResults = await resResp.json();
    edaData = await edaResp.json();

    updateHeaderStats();
    updateOverviewMetrics();
    renderModelComparisonChart();
    renderOverfittingChart();
    renderQuickForecastChart();
    renderFullPredictionsChart();
    renderResidualsCharts();
    renderEdaCharts();
    renderDiagnosticCharts();
    initSimulatorDefaults();
    loadDatasetTable();

  } catch (err) {
    console.error('Failed to load dashboard data:', err);
  } finally {
    showLoading(false);
  }
}

function showLoading(show, text = '') {
  const banner = document.getElementById('loadingBanner');
  const bannerText = document.getElementById('loadingText');
  if (banner) {
    if (show) {
      banner.classList.remove('hidden');
      if (bannerText && text) bannerText.textContent = text;
    } else {
      banner.classList.add('hidden');
    }
  }
}

// Update Top Header Quick Metrics
function updateHeaderStats() {
  if (!edaData || !edaData.price_stats) return;
  const stats = edaData.price_stats;

  const priceEl = document.getElementById('headerLatestPrice');
  if (priceEl) priceEl.textContent = `$${stats.latest.toFixed(2)}`;

  const deltaEl = document.getElementById('headerMomDelta');
  if (deltaEl) {
    const delta = stats.latest_mom_change;
    const pct = stats.latest_pct_change;
    const isUp = delta >= 0;
    deltaEl.className = `text-sm font-bold font-mono ${isUp ? 'text-emerald-400' : 'text-rose-400'}`;
    deltaEl.textContent = `${isUp ? '+' : ''}${delta.toFixed(2)} (${isUp ? '+' : ''}${pct.toFixed(1)}%)`;
  }

  const opecEl = document.getElementById('headerOpecStance');
  if (opecEl) {
    const opec = stats.latest_opec;
    const label = opec === 1 ? 'Increase' : opec === -1 ? 'Cut' : 'No Change';
    const color = opec === 1 ? 'text-emerald-400' : opec === -1 ? 'text-rose-400' : 'text-blue-400';
    opecEl.className = `text-sm font-semibold ${color}`;
    opecEl.textContent = label;
  }
}

// Update Overview Metric Cards
function updateOverviewMetrics() {
  if (!modelResults || !modelResults.metrics) return;
  const m = modelResults.metrics;
  const meta = modelResults.meta;
  const params = modelResults.best_params;

  // SVR
  document.getElementById('svrR2Val').textContent = m.svr.te_r2.toFixed(4);
  document.getElementById('svrMaeVal').textContent = `$${m.svr.te_mae.toFixed(2)}`;
  document.getElementById('svrExtraMetrics').textContent = `${m.svr.te_rmse.toFixed(2)} / ${m.svr.mape.toFixed(1)}%`;
  if (params && params.svr && params.svr.kernel) {
    document.getElementById('svrKernelBadge').textContent = params.svr.kernel.toUpperCase();
    document.getElementById('bestSvrParamsText').textContent = JSON.stringify(params.svr).replace(/[{""}]/g, '').replace(/:/g, '=');
  }

  // RF
  document.getElementById('rfR2Val').textContent = m.rf.te_r2.toFixed(4);
  document.getElementById('rfMaeVal').textContent = `$${m.rf.te_mae.toFixed(2)}`;
  document.getElementById('rfExtraMetrics').textContent = `${m.rf.te_rmse.toFixed(2)} / ${m.rf.mape.toFixed(1)}%`;
  if (params && params.rf && params.rf.n_estimators) {
    document.getElementById('rfTreesBadge').textContent = `${params.rf.n_estimators} Trees`;
    document.getElementById('bestRfParamsText').textContent = JSON.stringify(params.rf).replace(/[{""}]/g, '').replace(/:/g, '=');
  }

  // Naive
  document.getElementById('naiveR2Val').textContent = m.naive.te_r2.toFixed(4);
  document.getElementById('naiveMaeVal').textContent = `$${m.naive.te_mae.toFixed(2)}`;

  // Meta
  document.getElementById('totalRecordsVal').textContent = meta.total_records;
  document.getElementById('outliersVal').textContent = `${meta.outliers_count} (${((meta.outliers_count / meta.total_records) * 100).toFixed(1)}%)`;
  document.getElementById('splitBadge').textContent = `${meta.train_pct}% / ${meta.test_pct}%`;
  document.getElementById('dateRangeVal').textContent = `${meta.date_min} → ${meta.date_max}`;

  // Support Vectors
  if (params.support_vectors) {
    document.getElementById('supportVecCount').textContent = params.support_vectors.count;
    document.getElementById('supportVecPct').textContent = `${params.support_vectors.pct_of_train}% of train data`;
  }
}

// 1. Model Comparison Bar Chart
function renderModelComparisonChart() {
  const ctx = document.getElementById('modelComparisonChart');
  if (!ctx || !modelResults) return;

  if (charts.modelComparison) charts.modelComparison.destroy();

  const m = modelResults.metrics;

  charts.modelComparison = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['R² (Higher is better)', 'MAE (Lower is better)', 'RMSE (Lower is better)', 'MAPE % (Lower is better)'],
      datasets: [
        {
          label: 'SVR (Tuned)',
          data: [m.svr.te_r2, m.svr.te_mae, m.svr.te_rmse, m.svr.mape],
          backgroundColor: 'rgba(239, 68, 68, 0.85)',
          borderRadius: 6
        },
        {
          label: 'Random Forest',
          data: [m.rf.te_r2, m.rf.te_mae, m.rf.te_rmse, m.rf.mape],
          backgroundColor: 'rgba(16, 185, 129, 0.85)',
          borderRadius: 6
        },
        {
          label: 'Naive Baseline',
          data: [m.naive.te_r2, m.naive.te_mae, m.naive.te_rmse, m.naive.mape],
          backgroundColor: 'rgba(168, 85, 247, 0.85)',
          borderRadius: 6
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#0f172a',
          titleColor: '#e2e8f0',
          bodyColor: '#cbd5e1',
          borderColor: '#334155',
          borderWidth: 1,
          padding: 10
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(51, 65, 85, 0.3)' },
          ticks: { color: '#94a3b8', font: { size: 11 } }
        },
        y: {
          grid: { color: 'rgba(51, 65, 85, 0.3)' },
          ticks: { color: '#94a3b8', font: { size: 11 } }
        }
      }
    }
  });
}

// 2. Overfitting Check Chart
function renderOverfittingChart() {
  const ctx = document.getElementById('overfittingChart');
  if (!ctx || !modelResults) return;

  if (charts.overfitting) charts.overfitting.destroy();

  const m = modelResults.metrics;

  charts.overfitting = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['SVR Train', 'SVR Test', 'RF Train', 'RF Test'],
      datasets: [{
        data: [m.svr.tr_r2, m.svr.te_r2, m.rf.tr_r2, m.rf.te_r2],
        backgroundColor: [
          'rgba(239, 68, 68, 0.45)',
          'rgba(239, 68, 68, 0.95)',
          'rgba(16, 185, 129, 0.45)',
          'rgba(16, 185, 129, 0.95)'
        ],
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `R²: ${ctx.parsed.y.toFixed(4)}`
          }
        }
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: '#94a3b8', font: { size: 10 } }
        },
        y: {
          min: 0,
          max: 1.05,
          grid: { color: 'rgba(51, 65, 85, 0.3)' },
          ticks: { color: '#94a3b8', font: { size: 10 } }
        }
      }
    }
  });
}

// 3. Quick Forecast Chart (Overview Tab)
function renderQuickForecastChart() {
  const ctx = document.getElementById('quickForecastChart');
  if (!ctx || !modelResults) return;

  if (charts.quickForecast) charts.quickForecast.destroy();

  const preds = modelResults.test_predictions;

  charts.quickForecast = new Chart(ctx, {
    type: 'line',
    data: {
      labels: preds.dates,
      datasets: [
        {
          label: 'Actual Price',
          data: preds.actual,
          borderColor: '#6366f1',
          backgroundColor: 'rgba(99, 102, 241, 0.1)',
          fill: true,
          tension: 0.2,
          borderWidth: 2,
          pointRadius: 2.5
        },
        {
          label: 'SVR Predicted',
          data: preds.svr,
          borderColor: '#ef4444',
          borderDash: [5, 5],
          tension: 0.2,
          borderWidth: 2,
          pointRadius: 0
        },
        {
          label: 'RF Predicted',
          data: preds.rf,
          borderColor: '#10b981',
          borderDash: [3, 3],
          tension: 0.2,
          borderWidth: 2,
          pointRadius: 0
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          position: 'top',
          labels: { color: '#cbd5e1', font: { size: 11 }, boxWidth: 14 }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(51, 65, 85, 0.2)' },
          ticks: { color: '#94a3b8', font: { size: 10 }, maxTicksLimit: 10 }
        },
        y: {
          grid: { color: 'rgba(51, 65, 85, 0.3)' },
          ticks: { color: '#94a3b8', font: { size: 10 } }
        }
      }
    }
  });
}

// 4. Full Predictions Chart (Tab 2)
function renderFullPredictionsChart() {
  const ctx = document.getElementById('fullPredictionsChart');
  if (!ctx || !modelResults) return;

  if (charts.fullPredictions) charts.fullPredictions.destroy();

  const preds = modelResults.test_predictions;

  charts.fullPredictions = new Chart(ctx, {
    type: 'line',
    data: {
      labels: preds.dates,
      datasets: [
        {
          label: 'Actual Settlement Price',
          data: preds.actual,
          borderColor: '#6366f1',
          backgroundColor: 'rgba(99, 102, 241, 0.08)',
          fill: true,
          borderWidth: 2.5,
          tension: 0.2,
          pointRadius: 3
        },
        {
          label: 'SVR Predicted',
          data: preds.svr,
          borderColor: '#ef4444',
          borderDash: [5, 4],
          borderWidth: 2,
          tension: 0.2,
          pointRadius: 2
        },
        {
          label: 'Random Forest Predicted',
          data: preds.rf,
          borderColor: '#10b981',
          borderDash: [4, 4],
          borderWidth: 2,
          tension: 0.2,
          pointRadius: 2
        },
        {
          label: 'Naive Baseline [P(t-1)]',
          data: preds.naive,
          borderColor: '#a855f7',
          borderDash: [2, 2],
          borderWidth: 1.5,
          tension: 0.2,
          pointRadius: 0
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false }
      },
      scales: {
        x: {
          grid: { color: 'rgba(51, 65, 85, 0.2)' },
          ticks: { color: '#94a3b8', font: { size: 11 }, maxTicksLimit: 14 }
        },
        y: {
          title: { display: true, text: 'USD / bbl', color: '#94a3b8' },
          grid: { color: 'rgba(51, 65, 85, 0.3)' },
          ticks: { color: '#94a3b8', font: { size: 11 } }
        }
      }
    }
  });
}

function updatePredictionsSeries() {
  if (!charts.fullPredictions) return;
  charts.fullPredictions.getDatasetMeta(0).hidden = !document.getElementById('toggleActual').checked;
  charts.fullPredictions.getDatasetMeta(1).hidden = !document.getElementById('toggleSvr').checked;
  charts.fullPredictions.getDatasetMeta(2).hidden = !document.getElementById('toggleRf').checked;
  charts.fullPredictions.getDatasetMeta(3).hidden = !document.getElementById('toggleNaive').checked;
  charts.fullPredictions.update();
}

// 5. Residuals Charts
function renderResidualsCharts() {
  const preds = modelResults.test_predictions;
  const isSvr = currentResidualModel === 'svr';
  const residuals = isSvr ? preds.svr_residuals : preds.rf_residuals;
  const color = isSvr ? '#ef4444' : '#10b981';

  // Residuals over time
  const ctxTime = document.getElementById('residualsTimeChart');
  if (ctxTime) {
    if (charts.resTime) charts.resTime.destroy();

    charts.resTime = new Chart(ctxTime, {
      type: 'line',
      data: {
        labels: preds.dates,
        datasets: [{
          label: `${isSvr ? 'SVR' : 'RF'} Residual ($)`,
          data: residuals,
          borderColor: color,
          backgroundColor: `${color}15`,
          borderWidth: 1.8,
          pointRadius: 2,
          fill: true
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: {
            grid: { color: 'rgba(51, 65, 85, 0.2)' },
            ticks: { color: '#94a3b8', maxTicksLimit: 8 }
          },
          y: {
            grid: { color: 'rgba(51, 65, 85, 0.3)' },
            ticks: { color: '#94a3b8' }
          }
        }
      }
    });
  }

  // Residual Distribution Histogram
  const ctxHist = document.getElementById('residualHistChart');
  if (ctxHist) {
    if (charts.resHist) charts.resHist.destroy();

    // compute histogram of residuals
    const minR = Math.min(...residuals);
    const maxR = Math.max(...residuals);
    const binCount = 18;
    const step = (maxR - minR) / binCount;
    const bins = Array(binCount).fill(0);
    const labels = [];

    for (let i = 0; i < binCount; i++) {
      const bMin = minR + i * step;
      const bMax = bMin + step;
      labels.push(`${bMin.toFixed(1)} to ${bMax.toFixed(1)}`);
    }

    residuals.forEach(r => {
      let bIdx = Math.floor((r - minR) / step);
      if (bIdx >= binCount) bIdx = binCount - 1;
      if (bIdx >= 0) bins[bIdx]++;
    });

    const pctBins = bins.map(b => ((b / residuals.length) * 100).toFixed(1));

    // Stats
    const mean = residuals.reduce((a, b) => a + b, 0) / residuals.length;
    const variance = residuals.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / residuals.length;
    const std = Math.sqrt(variance);

    const badge = document.getElementById('residualStatsBadge');
    if (badge) badge.textContent = `Mean: ${mean.toFixed(2)} | Std: ${std.toFixed(2)}`;

    charts.resHist = new Chart(ctxHist, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Percentage of Observations (%)',
          data: pctBins,
          backgroundColor: `${color}cc`,
          borderRadius: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (c) => `${c.parsed.y}% of observations`
            }
          }
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: { color: '#94a3b8', font: { size: 9 }, maxRotation: 45 }
          },
          y: {
            title: { display: true, text: '% of Observations', color: '#94a3b8' },
            grid: { color: 'rgba(51, 65, 85, 0.3)' },
            ticks: { color: '#94a3b8', callback: v => `${v}%` }
          }
        }
      }
    });
  }
}

function setResidualModel(model) {
  currentResidualModel = model;
  const svrBtn = document.getElementById('resModelSvrBtn');
  const rfBtn = document.getElementById('resModelRfBtn');

  if (model === 'svr') {
    svrBtn.className = 'px-2.5 py-1 rounded bg-rose-500/20 text-rose-300 border border-rose-500/30';
    rfBtn.className = 'px-2.5 py-1 rounded bg-slate-800 text-slate-400 border border-slate-700';
  } else {
    rfBtn.className = 'px-2.5 py-1 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30';
    svrBtn.className = 'px-2.5 py-1 rounded bg-slate-800 text-slate-400 border border-slate-700';
  }
  renderResidualsCharts();
}

// 6. Market EDA Charts (Tab 3)
function renderEdaCharts() {
  if (!edaData) return;

  // A. Time Series with Outliers
  const ctxTs = document.getElementById('edaTimeSeriesChart');
  if (ctxTs) {
    if (charts.edaTs) charts.edaTs.destroy();

    const ts = edaData.timeseries;
    const labels = ts.map(t => t.date);
    const prices = ts.map(t => t.price);
    const outliers = ts.map(t => t.is_outlier ? t.price : null);

    charts.edaTs = new Chart(ctxTs, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Crude Oil Settlement Price',
            data: prices,
            borderColor: '#6366f1',
            borderWidth: 2,
            tension: 0.2,
            pointRadius: 0
          },
          {
            label: 'IQR Outlier Point',
            data: outliers,
            borderColor: '#ef4444',
            backgroundColor: '#ef4444',
            type: 'scatter',
            pointRadius: 5,
            pointHoverRadius: 7
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: {
            grid: { color: 'rgba(51, 65, 85, 0.2)' },
            ticks: { color: '#94a3b8', maxTicksLimit: 14 }
          },
          y: {
            grid: { color: 'rgba(51, 65, 85, 0.3)' },
            ticks: { color: '#94a3b8' }
          }
        }
      }
    });
  }

  // B. Price Distribution Histogram
  const ctxDist = document.getElementById('edaDistributionChart');
  if (ctxDist) {
    if (charts.edaDist) charts.edaDist.destroy();

    const bins = edaData.hist_bins;
    const labels = bins.map(b => `$${b.range}`);
    const counts = bins.map(b => b.count);

    if (edaData.price_stats) {
      const ps = edaData.price_stats;
      document.getElementById('priceDistributionStats').textContent =
        `Min: $${ps.min.toFixed(2)} | Mean: $${ps.mean.toFixed(2)} | Max: $${ps.max.toFixed(2)}`;
    }

    charts.edaDist = new Chart(ctxDist, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Frequency',
          data: counts,
          backgroundColor: 'rgba(59, 130, 246, 0.8)',
          borderRadius: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: {
            grid: { display: false },
            ticks: { color: '#94a3b8', font: { size: 9 }, maxRotation: 45 }
          },
          y: {
            grid: { color: 'rgba(51, 65, 85, 0.3)' },
            ticks: { color: '#94a3b8' }
          }
        }
      }
    });
  }

  // C. OPEC Policy Stance
  const ctxOpec = document.getElementById('edaOpecChart');
  if (ctxOpec) {
    if (charts.edaOpec) charts.edaOpec.destroy();

    const opec = edaData.opec_groups;
    const labels = ['Production Cut (-1)', 'No Change (0)', 'Production Increase (+1)'];
    const keys = ['Cut', 'No Change', 'Increase'];

    const means = keys.map(k => opec[k] ? opec[k].mean : 0);
    const medians = keys.map(k => opec[k] ? opec[k].median : 0);

    charts.edaOpec = new Chart(ctxOpec, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Mean Price ($/bbl)',
            data: means,
            backgroundColor: 'rgba(59, 130, 246, 0.8)',
            borderRadius: 6
          },
          {
            label: 'Median Price ($/bbl)',
            data: medians,
            backgroundColor: 'rgba(16, 185, 129, 0.8)',
            borderRadius: 6
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'top',
            labels: { color: '#cbd5e1', font: { size: 10 } }
          }
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: { color: '#94a3b8', font: { size: 10 } }
          },
          y: {
            grid: { color: 'rgba(51, 65, 85, 0.3)' },
            ticks: { color: '#94a3b8' }
          }
        }
      }
    });
  }

  // D. Demand vs Price Scatter
  const ctxDemand = document.getElementById('edaDemandScatterChart');
  if (ctxDemand) {
    if (charts.edaDemand) charts.edaDemand.destroy();

    const scatterData = edaData.timeseries.map(t => ({ x: t.demand, y: t.price }));

    charts.edaDemand = new Chart(ctxDemand, {
      type: 'scatter',
      data: {
        datasets: [{
          label: 'Demand vs Price',
          data: scatterData,
          backgroundColor: 'rgba(16, 185, 129, 0.65)',
          pointRadius: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: {
            title: { display: true, text: 'Demand (Mbbl/day)', color: '#94a3b8' },
            grid: { color: 'rgba(51, 65, 85, 0.2)' },
            ticks: { color: '#94a3b8' }
          },
          y: {
            title: { display: true, text: 'Price (USD/bbl)', color: '#94a3b8' },
            grid: { color: 'rgba(51, 65, 85, 0.3)' },
            ticks: { color: '#94a3b8' }
          }
        }
      }
    });
  }

  // E. USD Index vs Price Scatter
  const ctxUsd = document.getElementById('edaUsdScatterChart');
  if (ctxUsd) {
    if (charts.edaUsd) charts.edaUsd.destroy();

    const scatterData = edaData.timeseries.map(t => ({ x: t.usd, y: t.price }));

    charts.edaUsd = new Chart(ctxUsd, {
      type: 'scatter',
      data: {
        datasets: [{
          label: 'USD Index vs Price',
          data: scatterData,
          backgroundColor: 'rgba(245, 158, 11, 0.65)',
          pointRadius: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: {
            title: { display: true, text: 'USD Index (DXY)', color: '#94a3b8' },
            grid: { color: 'rgba(51, 65, 85, 0.2)' },
            ticks: { color: '#94a3b8' }
          },
          y: {
            title: { display: true, text: 'Price (USD/bbl)', color: '#94a3b8' },
            grid: { color: 'rgba(51, 65, 85, 0.3)' },
            ticks: { color: '#94a3b8' }
          }
        }
      }
    });
  }
}

// 7. Model Diagnostic Charts (Tab 4)
function renderDiagnosticCharts() {
  if (!modelResults) return;

  const fi = modelResults.feature_importance;
  const lc = modelResults.learning_curves;

  // A. SVR Importance Chart
  const ctxSvrImp = document.getElementById('svrImportanceChart');
  if (ctxSvrImp && fi && fi.svr) {
    if (charts.svrImp) charts.svrImp.destroy();

    const topSvr = fi.svr.slice(0, 10).reverse();
    const labels = topSvr.map(f => f.feature);
    const values = topSvr.map(f => f.importance);

    charts.svrImp = new Chart(ctxSvrImp, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Drop in R²',
          data: values,
          backgroundColor: 'rgba(239, 68, 68, 0.85)',
          borderRadius: 4
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: {
            grid: { color: 'rgba(51, 65, 85, 0.2)' },
            ticks: { color: '#94a3b8' }
          },
          y: {
            grid: { display: false },
            ticks: { color: '#94a3b8', font: { size: 10 } }
          }
        }
      }
    });
  }

  // B. RF Importance Chart
  const ctxRfImp = document.getElementById('rfImportanceChart');
  if (ctxRfImp && fi && fi.rf) {
    if (charts.rfImp) charts.rfImp.destroy();

    const topRf = fi.rf.slice(0, 10).reverse();
    const labels = topRf.map(f => f.feature);
    const values = topRf.map(f => f.importance);

    charts.rfImp = new Chart(ctxRfImp, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Impurity Score',
          data: values,
          backgroundColor: 'rgba(16, 185, 129, 0.85)',
          borderRadius: 4
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: {
            grid: { color: 'rgba(51, 65, 85, 0.2)' },
            ticks: { color: '#94a3b8' }
          },
          y: {
            grid: { display: false },
            ticks: { color: '#94a3b8', font: { size: 10 } }
          }
        }
      }
    });
  }

  // C. SVR Learning Curve
  const ctxSvrLc = document.getElementById('svrLearningCurveChart');
  if (ctxSvrLc && lc && lc.svr) {
    if (charts.svrLc) charts.svrLc.destroy();

    charts.svrLc = new Chart(ctxSvrLc, {
      type: 'line',
      data: {
        labels: lc.svr.sizes.map(s => `${s} samples`),
        datasets: [
          {
            label: 'Train R²',
            data: lc.svr.train,
            borderColor: '#ef4444',
            borderWidth: 2,
            pointRadius: 4
          },
          {
            label: 'Validation R²',
            data: lc.svr.val,
            borderColor: '#fca5a5',
            borderDash: [5, 5],
            borderWidth: 2,
            pointRadius: 4
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: { display: true, text: 'SVR Learning Curve', color: '#cbd5e1' },
          legend: { labels: { color: '#94a3b8', font: { size: 10 } } }
        },
        scales: {
          x: {
            grid: { color: 'rgba(51, 65, 85, 0.2)' },
            ticks: { color: '#94a3b8', font: { size: 10 } }
          },
          y: {
            min: -0.2,
            max: 1.05,
            grid: { color: 'rgba(51, 65, 85, 0.3)' },
            ticks: { color: '#94a3b8' }
          }
        }
      }
    });
  }

  // D. RF Learning Curve
  const ctxRfLc = document.getElementById('rfLearningCurveChart');
  if (ctxRfLc && lc && lc.rf) {
    if (charts.rfLc) charts.rfLc.destroy();

    charts.rfLc = new Chart(ctxRfLc, {
      type: 'line',
      data: {
        labels: lc.rf.sizes.map(s => `${s} samples`),
        datasets: [
          {
            label: 'Train R²',
            data: lc.rf.train,
            borderColor: '#10b981',
            borderWidth: 2,
            pointRadius: 4
          },
          {
            label: 'Validation R²',
            data: lc.rf.val,
            borderColor: '#6ee7b7',
            borderDash: [5, 5],
            borderWidth: 2,
            pointRadius: 4
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: { display: true, text: 'Random Forest Learning Curve', color: '#cbd5e1' },
          legend: { labels: { color: '#94a3b8', font: { size: 10 } } }
        },
        scales: {
          x: {
            grid: { color: 'rgba(51, 65, 85, 0.2)' },
            ticks: { color: '#94a3b8', font: { size: 10 } }
          },
          y: {
            min: -0.2,
            max: 1.05,
            grid: { color: 'rgba(51, 65, 85, 0.3)' },
            ticks: { color: '#94a3b8' }
          }
        }
      }
    });
  }

  // E. Decision Tree Rules Table
  const treeTableBody = document.getElementById('treeRulesTableBody');
  if (treeTableBody && modelResults.tree_preview) {
    treeTableBody.innerHTML = '';
    modelResults.tree_preview.forEach(rule => {
      const tr = document.createElement('tr');
      tr.className = 'hover:bg-slate-900/60 transition';
      tr.innerHTML = `
        <td class="py-2 px-2 text-slate-500">#${rule.node}</td>
        <td class="py-2 px-2 text-emerald-400 font-semibold">${rule.feature} &le; ${rule.threshold}</td>
        <td class="py-2 px-2 text-slate-300">${rule.samples}</td>
        <td class="py-2 px-2 text-white font-bold">$${rule.value.toFixed(2)}</td>
      `;
      treeTableBody.appendChild(tr);
    });
  }
}

// 8. Scenario Simulator (Tab 5)
function initSimulatorDefaults() {
  if (!edaData || !edaData.price_stats) return;
  const s = edaData.price_stats;

  document.getElementById('simDemand').value = s.latest_demand;
  document.getElementById('simDemandVal').textContent = s.latest_demand.toFixed(1);

  document.getElementById('simUsd').value = s.latest_usd;
  document.getElementById('simUsdVal').textContent = s.latest_usd.toFixed(1);

  document.getElementById('simLag1').value = s.latest;
  document.getElementById('simLag1Val').textContent = `$${s.latest.toFixed(2)}`;

  document.getElementById('simLag2').value = (s.latest - s.latest_mom_change).toFixed(2);
  document.getElementById('simLag3').value = (s.latest - s.latest_mom_change).toFixed(2);

  setSimOpec(s.latest_opec);
  runSimulation();
}

function resetSimulator() {
  initSimulatorDefaults();
}

function setSimOpec(val) {
  currentSimOpec = val;
  const cutBtn = document.getElementById('simOpecCutBtn');
  const noBtn = document.getElementById('simOpecNoBtn');
  const incBtn = document.getElementById('simOpecIncBtn');

  [cutBtn, noBtn, incBtn].forEach(b => {
    b.className = 'py-2 px-3 rounded-lg text-xs font-semibold border border-slate-700 bg-slate-800 text-slate-300 transition';
  });

  if (val === -1) {
    cutBtn.className = 'py-2 px-3 rounded-lg text-xs font-semibold border border-rose-500/50 bg-rose-500/20 text-rose-300';
  } else if (val === 1) {
    incBtn.className = 'py-2 px-3 rounded-lg text-xs font-semibold border border-emerald-500/50 bg-emerald-500/20 text-emerald-300';
  } else {
    noBtn.className = 'py-2 px-3 rounded-lg text-xs font-semibold border border-blue-500/50 bg-blue-500/20 text-blue-300';
  }

  runSimulation();
}

async function runSimulation() {
  const demand = parseFloat(document.getElementById('simDemand').value);
  const usd = parseFloat(document.getElementById('simUsd').value);
  const lag1 = parseFloat(document.getElementById('simLag1').value);
  const lag2 = parseFloat(document.getElementById('simLag2').value) || lag1;
  const lag3 = parseFloat(document.getElementById('simLag3').value) || lag2;
  const month = parseInt(document.getElementById('simMonth').value) || 10;

  document.getElementById('simDemandVal').textContent = demand.toFixed(1);
  document.getElementById('simUsdVal').textContent = usd.toFixed(1);
  document.getElementById('simLag1Val').textContent = `$${lag1.toFixed(2)}`;

  try {
    const resp = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        demand: demand,
        usd_index: usd,
        opec_policy: currentSimOpec,
        lag1: lag1,
        lag2: lag2,
        lag3: lag3,
        month: month,
        year: 2024
      })
    });

    const res = await resp.json();

    document.getElementById('simEnsemblePrice').textContent = `$${res.ensemble_avg.toFixed(2)}`;
    document.getElementById('simSvrPrice').textContent = `$${res.svr_predicted_price.toFixed(2)}`;
    document.getElementById('simRfPrice').textContent = `$${res.rf_predicted_price.toFixed(2)}`;
    document.getElementById('simNaivePrice').textContent = `$${res.naive_benchmark.toFixed(2)}`;

    const delta = res.ensemble_avg - lag1;
    const isUp = delta >= 0;
    const deltaEl = document.getElementById('simDeltaVsLag1');
    deltaEl.className = `text-xs font-mono mt-1 ${isUp ? 'text-emerald-400' : 'text-rose-400'}`;
    deltaEl.textContent = `Δ ${isUp ? '+' : ''}${delta.toFixed(2)} vs Previous Month (${isUp ? '+' : ''}${((delta / lag1) * 100).toFixed(1)}%)`;

  } catch (err) {
    console.error('Simulation error:', err);
  }
}

// 9. Dataset Explorer Table (Tab 6)
async function loadDatasetTable() {
  const searchVal = document.getElementById('tableSearchInput')?.value || '';
  const url = `/api/data?page=${tableCurrentPage}&page_size=50&search=${encodeURIComponent(searchVal)}`;

  try {
    const resp = await fetch(url);
    const data = await resp.json();

    const thead = document.getElementById('datasetTableHead');
    const tbody = document.getElementById('datasetTableBody');

    // Build headers
    thead.innerHTML = '';
    const htr = document.createElement('tr');
    data.columns.forEach(col => {
      const th = document.createElement('th');
      th.className = 'py-3 px-3';
      th.textContent = col;
      htr.appendChild(th);
    });
    thead.appendChild(htr);

    // Build rows
    tbody.innerHTML = '';
    if (data.records.length === 0) {
      tbody.innerHTML = '<tr><td colspan="15" class="py-4 text-center text-slate-500">No matching records found</td></tr>';
      return;
    }

    data.records.forEach(row => {
      const tr = document.createElement('tr');
      tr.className = 'hover:bg-slate-900/60 transition';

      data.columns.forEach(col => {
        const td = document.createElement('td');
        td.className = 'py-2 px-3 whitespace-nowrap';
        const val = row[col];

        if (col === 'Is_Outlier' && val === true) {
          td.innerHTML = '<span class="px-1.5 py-0.5 rounded bg-rose-500/20 text-rose-400 text-[10px] font-bold">OUTLIER</span>';
        } else if (col.includes('Residual') && typeof val === 'number') {
          const color = val >= 0 ? 'text-emerald-400' : 'text-rose-400';
          td.className += ` ${color}`;
          td.textContent = (val >= 0 ? '+' : '') + val.toFixed(2);
        } else if (col.includes('Price') && typeof val === 'number') {
          td.className += ' font-bold text-white';
          td.textContent = `$${val.toFixed(2)}`;
        } else {
          td.textContent = val !== null && val !== undefined ? val : '-';
        }

        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });

    // Pagination info
    const totalPages = Math.ceil(data.total / data.page_size) || 1;
    document.getElementById('tablePaginationInfo').textContent =
      `Showing ${(tableCurrentPage - 1) * data.page_size + 1} to ${Math.min(tableCurrentPage * data.page_size, data.total)} of ${data.total} records`;
    document.getElementById('currentPageNumber').textContent = tableCurrentPage;
    document.getElementById('prevPageBtn').disabled = tableCurrentPage <= 1;
    document.getElementById('nextPageBtn').disabled = tableCurrentPage >= totalPages;

  } catch (err) {
    console.error('Failed to load dataset table:', err);
  }
}

function debounceSearch() {
  clearTimeout(searchDebounceTimer);
  searchDebounceTimer = setTimeout(() => {
    tableCurrentPage = 1;
    loadDatasetTable();
  }, 300);
}

function prevTablePage() {
  if (tableCurrentPage > 1) {
    tableCurrentPage--;
    loadDatasetTable();
  }
}

function nextTablePage() {
  tableCurrentPage++;
  loadDatasetTable();
}

// 10. File Upload & Retraining (Tab 7)
function setupDragAndDrop() {
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');

  if (!dropZone || !fileInput) return;

  dropZone.addEventListener('click', () => fileInput.click());

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('border-blue-500', 'bg-blue-500/10');
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('border-blue-500', 'bg-blue-500/10');
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('border-blue-500', 'bg-blue-500/10');
    if (e.dataTransfer.files.length) {
      fileInput.files = e.dataTransfer.files;
      fileSelected({ target: fileInput });
    }
  });
}

function fileSelected(e) {
  const file = e.target.files[0];
  if (file) {
    document.getElementById('uploadFileName').textContent = `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  }
}

async function handleFileUpload(e) {
  e.preventDefault();
  const fileInput = document.getElementById('fileInput');
  if (!fileInput.files || fileInput.files.length === 0) {
    alert('Please select a file to upload.');
    return;
  }

  const formData = new FormData();
  formData.append('file', fileInput.files[0]);

  showLoading(true, 'Uploading dataset and running model retraining pipeline...');
  try {
    const resp = await fetch('/api/upload', {
      method: 'POST',
      body: formData
    });

    const res = await resp.json();
    if (!resp.ok) throw new Error(res.detail || 'Upload failed');

    alert(`Successfully processed ${res.records} rows! Dashboard updated.`);
    await loadDashboardData();
    switchTab('tab-overview');
  } catch (err) {
    alert(`Error: ${err.message}`);
  } finally {
    showLoading(false);
  }
}

async function triggerRetrain() {
  const split = parseFloat(document.getElementById('testSplitInput').value);
  const tuneIter = parseInt(document.getElementById('tuneIterInput').value);

  showLoading(true, `Retraining SVR and Random Forest models (Split: ${Math.round((1 - split) * 100)}/${Math.round(split * 100)}, Iterations: ${tuneIter})...`);
  try {
    const resp = await fetch(`/api/retrain?test_split=${split}&tune_iter=${tuneIter}`, {
      method: 'POST'
    });

    const res = await resp.json();
    if (!resp.ok) throw new Error(res.detail || 'Retraining failed');

    await loadDashboardData();
    switchTab('tab-overview');
  } catch (err) {
    alert(`Error during retraining: ${err.message}`);
  } finally {
    showLoading(false);
  }
}
