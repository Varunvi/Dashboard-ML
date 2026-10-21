import os
import random
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore')

os.environ['PYTHONHASHSEED'] = '42'
random.seed(42)
np.random.seed(42)

FEATURES = [
    'Demand (Mbbl/day)', 'USD Index', 'OPEC Policy',
    'Month', 'Year', 'Quarter',
    'Price_Lag1', 'Price_Lag2', 'Price_Lag3', 'Price_Lag6',
    'Rolling_Mean_3', 'Rolling_Mean_5', 'Rolling_Mean_12',
    'Rolling_Std_5', 'Rolling_Std_12',
    'Price_MoM_Change', 'Price_3M_Change', 'Price_Pct_Change1',
    'Demand_x_OPEC', 'USD_x_Demand',
]


class OilModelEngine:
    def __init__(self, data_path="Oil prices 2.xlsx"):
        self.data_path = data_path
        self.df_raw = None
        self.df = None
        self.results = {}
        self.is_trained = False
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.best_svr = None
        self.best_rf = None

    def load_and_preprocess(self, file_path=None):
        path = file_path or self.data_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        if path.endswith('.csv'):
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)

        df.columns = df.columns.str.strip()
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.sort_values('Date').reset_index(drop=True)

        for col in ['Demand (Mbbl/day)', 'USD Index', 'Price (USD/bbl)']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'Demand (Mbbl/day)' in df.columns and 'USD Index' in df.columns:
            df[['Demand (Mbbl/day)', 'USD Index']] = (
                df[['Demand (Mbbl/day)', 'USD Index']]
                .interpolate(method='linear')
                .bfill().ffill()
            )

        if 'OPEC Policy' in df.columns:
            df['OPEC Policy'] = df['OPEC Policy'].fillna('No Change')
            if df['OPEC Policy'].dtype == object:
                df['OPEC Policy'] = df['OPEC Policy'].map(
                    {'Increase': 1, 'No Change': 0, 'Cut': -1, 1: 1, 0: 0, -1: -1}
                ).fillna(0)

        df = df.dropna(subset=['Price (USD/bbl)']).reset_index(drop=True)
        self.df_raw = df.copy()

        # Feature Engineering
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year
        df['Quarter'] = df['Date'].dt.quarter

        df['Price_Lag1'] = df['Price (USD/bbl)'].shift(1)
        df['Price_Lag2'] = df['Price (USD/bbl)'].shift(2)
        df['Price_Lag3'] = df['Price (USD/bbl)'].shift(3)
        df['Price_Lag6'] = df['Price (USD/bbl)'].shift(6)

        df['Rolling_Mean_3'] = df['Price (USD/bbl)'].rolling(window=3).mean()
        df['Rolling_Mean_5'] = df['Price (USD/bbl)'].rolling(window=5).mean()
        df['Rolling_Mean_12'] = df['Price (USD/bbl)'].rolling(window=12).mean()
        df['Rolling_Std_5'] = df['Price (USD/bbl)'].rolling(window=5).std()
        df['Rolling_Std_12'] = df['Price (USD/bbl)'].rolling(window=12).std()

        df['Price_MoM_Change'] = df['Price (USD/bbl)'].diff(1)
        df['Price_3M_Change'] = df['Price (USD/bbl)'].diff(3)
        df['Price_Pct_Change1'] = df['Price (USD/bbl)'].pct_change(1)

        df['Demand_x_OPEC'] = df['Demand (Mbbl/day)'] * df['OPEC Policy']
        df['USD_x_Demand'] = df['USD Index'] * df['Demand (Mbbl/day)']

        df = df.dropna().reset_index(drop=True)

        Q1, Q3 = df['Price (USD/bbl)'].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        outlier_mask = (
            (df['Price (USD/bbl)'] < Q1 - 1.5 * IQR) |
            (df['Price (USD/bbl)'] > Q3 + 1.5 * IQR)
        )
        df['Is_Outlier'] = outlier_mask

        self.df = df
        return df

    def train_models(self, test_split=0.30, tune_iter=15):
        if self.df is None or len(self.df) == 0:
            self.load_and_preprocess()

        df = self.df
        X = df[FEATURES]
        y = df['Price (USD/bbl)']

        split_idx = int((1.0 - test_split) * len(df))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        dates_train = df['Date'].iloc[:split_idx]
        dates_test = df['Date'].iloc[split_idx:]

        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

        X_train_sc = self.scaler_X.fit_transform(X_train)
        X_test_sc = self.scaler_X.transform(X_test)
        y_train_sc = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        y_test_sc = self.scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

        # Naive Baseline
        naive_pred = df.loc[X_test.index, 'Price_Lag1'].values
        naive_mae = mean_absolute_error(y_test, naive_pred)
        naive_rmse = np.sqrt(mean_squared_error(y_test, naive_pred))
        naive_r2 = r2_score(y_test, naive_pred)
        naive_mape = np.mean(np.abs((y_test.values - naive_pred) / y_test.values)) * 100

        # Baseline SVR & RF
        svr_base = SVR(kernel='rbf')
        svr_base.fit(X_train_sc, y_train_sc)
        svr_base_pred = self.scaler_y.inverse_transform(
            svr_base.predict(X_test_sc).reshape(-1, 1)).ravel()
        svr_base_r2 = r2_score(y_test, svr_base_pred)

        rf_base = RandomForestRegressor(random_state=42, n_estimators=100)
        rf_base.fit(X_train, y_train)
        rf_base_pred = rf_base.predict(X_test)
        rf_base_r2 = r2_score(y_test, rf_base_pred)

        # Hyperparameter Tuning
        tscv = TimeSeriesSplit(n_splits=5)

        param_dist_svr = {
            'kernel': ['rbf', 'linear'],
            'C': [1, 10, 50, 100],
            'gamma': ['scale', 0.01, 0.001],
            'epsilon': [0.1, 0.5, 1.0],
        }
        grid_svr = RandomizedSearchCV(
            SVR(), param_dist_svr,
            n_iter=min(tune_iter, 16),
            cv=tscv,
            scoring='neg_mean_squared_error',
            random_state=42, n_jobs=1
        )
        grid_svr.fit(X_train_sc, y_train_sc)
        self.best_svr = grid_svr.best_estimator_

        param_dist_rf = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2'],
        }
        grid_rf = RandomizedSearchCV(
            RandomForestRegressor(random_state=42),
            param_dist_rf,
            n_iter=min(tune_iter, 12),
            cv=tscv,
            scoring='neg_mean_squared_error',
            random_state=42, n_jobs=1
        )
        grid_rf.fit(X_train, y_train)
        self.best_rf = grid_rf.best_estimator_

        # Predictions
        svr_train_pred = self.scaler_y.inverse_transform(
            self.best_svr.predict(X_train_sc).reshape(-1, 1)).ravel()
        svr_test_pred = self.scaler_y.inverse_transform(
            self.best_svr.predict(X_test_sc).reshape(-1, 1)).ravel()

        rf_train_pred = self.best_rf.predict(X_train)
        rf_test_pred = self.best_rf.predict(X_test)

        # Evaluation
        def calc_metrics(y_tr, y_tr_p, y_te, y_te_p):
            tr_r2 = float(r2_score(y_tr, y_tr_p))
            te_r2 = float(r2_score(y_te, y_te_p))
            te_mae = float(mean_absolute_error(y_te, y_te_p))
            te_rmse = float(np.sqrt(mean_squared_error(y_te, y_te_p)))
            mape = float(np.mean(np.abs((np.array(y_te) - np.array(y_te_p)) / np.array(y_te))) * 100)
            return dict(tr_r2=tr_r2, te_r2=te_r2, te_mae=te_mae, te_rmse=te_rmse, mape=mape)

        svr_metrics = calc_metrics(y_train, svr_train_pred, y_test, svr_test_pred)
        rf_metrics = calc_metrics(y_train, rf_train_pred, y_test, rf_test_pred)
        naive_metrics = dict(
            tr_r2=float(r2_score(y_train.iloc[1:], df.loc[X_train.index[1:], 'Price_Lag1'])),
            te_r2=float(naive_r2),
            te_mae=float(naive_mae),
            te_rmse=float(naive_rmse),
            mape=float(naive_mape)
        )

        # Feature Importance
        perm = permutation_importance(
            self.best_svr, X_test_sc, y_test_sc,
            n_repeats=10, random_state=42, scoring='r2'
        )
        svr_feat_imp = [
            {"feature": feat, "importance": round(float(imp), 4), "std": round(float(std), 4)}
            for feat, imp, std in sorted(
                zip(FEATURES, perm.importances_mean, perm.importances_std),
                key=lambda x: x[1], reverse=True
            )
        ]

        rf_feat_imp = [
            {"feature": feat, "importance": round(float(imp), 4)}
            for feat, imp in sorted(
                zip(FEATURES, self.best_rf.feature_importances_),
                key=lambda x: x[1], reverse=True
            )
        ]

        # Learning Curves
        train_sizes = np.linspace(0.2, 1.0, 7)
        svr_learning = {"sizes": [], "train": [], "val": []}
        rf_learning = {"sizes": [], "train": [], "val": []}

        for size in train_sizes:
            n = max(int(size * len(X_train)), 10)
            X_tr, y_tr = X_train.iloc[:n], y_train.iloc[:n]
            X_va, y_va = (X_train.iloc[n:], y_train.iloc[n:]) if n < len(X_train) else (X_tr, y_tr)

            # SVR
            sX = StandardScaler()
            sy = StandardScaler()
            X_tr_s = sX.fit_transform(X_tr)
            y_tr_s = sy.fit_transform(y_tr.values.reshape(-1, 1)).ravel()
            X_va_s = sX.transform(X_va)

            temp_svr = SVR(**grid_svr.best_params_)
            temp_svr.fit(X_tr_s, y_tr_s)
            tr_p_s = sy.inverse_transform(temp_svr.predict(X_tr_s).reshape(-1, 1)).ravel()
            va_p_s = sy.inverse_transform(temp_svr.predict(X_va_s).reshape(-1, 1)).ravel()

            svr_learning["sizes"].append(int(n))
            svr_learning["train"].append(round(float(r2_score(y_tr, tr_p_s)), 4))
            svr_learning["val"].append(round(float(r2_score(y_va, va_p_s)), 4) if len(y_va) > 1 else None)

            # RF
            temp_rf = RandomForestRegressor(**grid_rf.best_params_, random_state=42)
            temp_rf.fit(X_tr, y_tr)
            tr_p_rf = temp_rf.predict(X_tr)
            va_p_rf = temp_rf.predict(X_va)

            rf_learning["sizes"].append(int(n))
            rf_learning["train"].append(round(float(r2_score(y_tr, tr_p_rf)), 4))
            rf_learning["val"].append(round(float(r2_score(y_va, va_p_rf)), 4) if len(y_va) > 1 else None)

        # Residuals
        svr_residuals = (y_test.values - svr_test_pred).tolist()
        rf_residuals = (y_test.values - rf_test_pred).tolist()

        # Update df with test predictions & residuals
        df.loc[X_test.index, 'SVR_Prediction'] = svr_test_pred
        df.loc[X_test.index, 'RF_Prediction'] = rf_test_pred
        df.loc[X_test.index, 'Naive_Prediction'] = naive_pred
        df.loc[X_test.index, 'SVR_Residual'] = svr_residuals
        df.loc[X_test.index, 'RF_Residual'] = rf_residuals

        # Save to final predictions Excel
        df.to_excel("final_predictions_improved.xlsx", index=False)

        # Support vectors information
        support_vec_count = int(len(self.best_svr.support_))
        total_train_samples = int(len(X_train))

        # Decision Tree preview
        tree_preview = []
        if len(self.best_rf.estimators_) > 0:
            tree = self.best_rf.estimators_[0].tree_
            # Extract top node splits
            for node_idx in range(min(15, tree.node_count)):
                if tree.feature[node_idx] >= 0:
                    tree_preview.append({
                        "node": node_idx,
                        "feature": FEATURES[tree.feature[node_idx]],
                        "threshold": round(float(tree.threshold[node_idx]), 2),
                        "samples": int(tree.n_node_samples[node_idx]),
                        "value": round(float(tree.value[node_idx][0][0]), 2)
                    })

        # Save results object
        self.results = {
            "meta": {
                "total_records": len(df),
                "train_records": len(X_train),
                "test_records": len(X_test),
                "train_pct": round(len(X_train) / len(df) * 100, 1),
                "test_pct": round(len(X_test) / len(df) * 100, 1),
                "date_min": df['Date'].min().strftime('%Y-%m-%d'),
                "date_max": df['Date'].max().strftime('%Y-%m-%d'),
                "outliers_count": int(df['Is_Outlier'].sum()),
                "features_count": len(FEATURES),
                "features": FEATURES,
            },
            "best_params": {
                "svr": grid_svr.best_params_,
                "rf": grid_rf.best_params_,
                "svr_base_r2": round(float(svr_base_r2), 4),
                "rf_base_r2": round(float(rf_base_r2), 4),
                "support_vectors": {
                    "count": support_vec_count,
                    "pct_of_train": round(support_vec_count / total_train_samples * 100, 1)
                }
            },
            "metrics": {
                "svr": svr_metrics,
                "rf": rf_metrics,
                "naive": naive_metrics,
                "improvements": {
                    "svr_r2_gain": round(svr_metrics['te_r2'] - naive_metrics['te_r2'], 4),
                    "rf_r2_gain": round(rf_metrics['te_r2'] - naive_metrics['te_r2'], 4),
                    "svr_mae_reduction": round(naive_metrics['te_mae'] - svr_metrics['te_mae'], 4),
                    "rf_mae_reduction": round(naive_metrics['te_mae'] - rf_metrics['te_mae'], 4),
                }
            },
            "test_predictions": {
                "dates": [d.strftime('%Y-%m-%d') for d in dates_test],
                "actual": [round(float(v), 2) for v in y_test.values],
                "svr": [round(float(v), 2) for v in svr_test_pred],
                "rf": [round(float(v), 2) for v in rf_test_pred],
                "naive": [round(float(v), 2) for v in naive_pred],
                "svr_residuals": [round(float(v), 2) for v in svr_residuals],
                "rf_residuals": [round(float(v), 2) for v in rf_residuals],
            },
            "feature_importance": {
                "svr": svr_feat_imp,
                "rf": rf_feat_imp,
            },
            "learning_curves": {
                "svr": svr_learning,
                "rf": rf_learning,
            },
            "tree_preview": tree_preview,
            "correlation_matrix": self._get_correlation_matrix(),
            "eda_summary": self._get_eda_summary()
        }

        self.is_trained = True
        return self.results

    def _get_correlation_matrix(self):
        corr = self.df[FEATURES + ['Price (USD/bbl)']].corr()
        return {
            "cols": list(corr.columns),
            "values": [[round(float(v), 3) if not np.isnan(v) else 0 for v in row] for row in corr.values]
        }

    def _get_eda_summary(self):
        df = self.df
        # OPEC boxplot grouping
        opec_groups = {}
        for code, label in [(-1, 'Cut'), (0, 'No Change'), (1, 'Increase')]:
            prices = df[df['OPEC Policy'] == code]['Price (USD/bbl)'].dropna().tolist()
            if prices:
                opec_groups[label] = {
                    "min": round(float(np.min(prices)), 2),
                    "q1": round(float(np.percentile(prices, 25)), 2),
                    "median": round(float(np.median(prices)), 2),
                    "q3": round(float(np.percentile(prices, 75)), 2),
                    "max": round(float(np.max(prices)), 2),
                    "mean": round(float(np.mean(prices)), 2),
                    "count": len(prices)
                }

        # Price distribution histogram bins
        hist, bin_edges = np.histogram(df['Price (USD/bbl)'], bins=20)
        hist_bins = [
            {"range": f"{round(bin_edges[i], 1)}-{round(bin_edges[i+1], 1)}",
             "mid": round((bin_edges[i] + bin_edges[i+1]) / 2, 1),
             "count": int(hist[i])}
            for i in range(len(hist))
        ]

        # Time series points with outlier flag
        ts_points = [
            {
                "date": row['Date'].strftime('%Y-%m-%d'),
                "price": round(float(row['Price (USD/bbl)']), 2),
                "demand": round(float(row['Demand (Mbbl/day)']), 2),
                "usd": round(float(row['USD Index']), 2),
                "opec": int(row['OPEC Policy']),
                "is_outlier": bool(row['Is_Outlier'])
            }
            for _, row in df.iterrows()
        ]

        return {
            "opec_groups": opec_groups,
            "hist_bins": hist_bins,
            "timeseries": ts_points,
            "price_stats": {
                "min": round(float(df['Price (USD/bbl)'].min()), 2),
                "max": round(float(df['Price (USD/bbl)'].max()), 2),
                "mean": round(float(df['Price (USD/bbl)'].mean()), 2),
                "std": round(float(df['Price (USD/bbl)'].std()), 2),
                "latest": round(float(df['Price (USD/bbl)'].iloc[-1]), 2),
                "latest_date": df['Date'].iloc[-1].strftime('%Y-%m-%d'),
                "latest_demand": round(float(df['Demand (Mbbl/day)'].iloc[-1]), 2),
                "latest_usd": round(float(df['USD Index'].iloc[-1]), 2),
                "latest_opec": int(df['OPEC Policy'].iloc[-1]),
                "latest_mom_change": round(float(df['Price_MoM_Change'].iloc[-1]), 2),
                "latest_pct_change": round(float(df['Price_Pct_Change1'].iloc[-1] * 100), 2)
            }
        }

    def predict_scenario(self, demand, usd_index, opec_policy, lag1, lag2=None, lag3=None, lag6=None, month=10, year=2024):
        if not self.is_trained:
            self.train_models()

        lag2 = lag2 if lag2 is not None else lag1
        lag3 = lag3 if lag3 is not None else lag2
        lag6 = lag6 if lag6 is not None else lag3
        quarter = (month - 1) // 3 + 1

        recent_lags = [lag1, lag2, lag3]
        rolling_3 = float(np.mean(recent_lags))
        rolling_5 = float(np.mean(recent_lags + [lag6, lag6]))
        rolling_12 = rolling_5
        rolling_std_5 = float(np.std(recent_lags + [lag6, lag6]))
        rolling_std_12 = rolling_std_5

        mom_change = lag1 - lag2
        m3_change = lag1 - lag3
        pct_change1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0.0

        demand_x_opec = demand * opec_policy
        usd_x_demand = usd_index * demand

        feat_dict = {
            'Demand (Mbbl/day)': demand,
            'USD Index': usd_index,
            'OPEC Policy': opec_policy,
            'Month': month,
            'Year': year,
            'Quarter': quarter,
            'Price_Lag1': lag1,
            'Price_Lag2': lag2,
            'Price_Lag3': lag3,
            'Price_Lag6': lag6,
            'Rolling_Mean_3': rolling_3,
            'Rolling_Mean_5': rolling_5,
            'Rolling_Mean_12': rolling_12,
            'Rolling_Std_5': rolling_std_5,
            'Rolling_Std_12': rolling_std_12,
            'Price_MoM_Change': mom_change,
            'Price_3M_Change': m3_change,
            'Price_Pct_Change1': pct_change1,
            'Demand_x_OPEC': demand_x_opec,
            'USD_x_Demand': usd_x_demand,
        }

        row_df = pd.DataFrame([feat_dict])[FEATURES]
        row_sc = self.scaler_X.transform(row_df)

        svr_pred_sc = self.best_svr.predict(row_sc).reshape(-1, 1)
        svr_val = float(self.scaler_y.inverse_transform(svr_pred_sc).ravel()[0])

        rf_val = float(self.best_rf.predict(row_df)[0])
        naive_val = float(lag1)

        return {
            "svr_predicted_price": round(svr_val, 2),
            "rf_predicted_price": round(rf_val, 2),
            "naive_benchmark": round(naive_val, 2),
            "ensemble_avg": round((svr_val + rf_val) / 2, 2),
            "scenario_inputs": feat_dict
        }


# Singleton engine instance
engine = OilModelEngine()

if __name__ == "__main__":
    res = engine.train_models()
    print("Train completed successfully!")
    print("SVR Test R2:", res['metrics']['svr']['te_r2'])
    print("RF  Test R2:", res['metrics']['rf']['te_r2'])
    print("Naive Test R2:", res['metrics']['naive']['te_r2'])
