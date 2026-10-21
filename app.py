import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np

from engine import engine, FEATURES

app = FastAPI(title="Crude Oil Price Prediction & Diagnostic Dashboard")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


class PredictRequest(BaseModel):
    demand: float
    usd_index: float
    opec_policy: int  # -1 (Cut), 0 (No Change), 1 (Increase)
    lag1: float
    lag2: Optional[float] = None
    lag3: Optional[float] = None
    lag6: Optional[float] = None
    month: Optional[int] = 10
    year: Optional[int] = 2024


@app.on_event("startup")
def startup_event():
    # Pre-train or initialize engine
    try:
        if not engine.is_trained:
            engine.train_models()
            print("Engine initialized and pre-trained successfully.")
    except Exception as e:
        print(f"Warning during engine initialization: {e}")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/api/status")
def get_status():
    return {
        "is_trained": engine.is_trained,
        "has_data": engine.df is not None,
        "records_count": len(engine.df) if engine.df is not None else 0
    }


@app.get("/api/results")
def get_results():
    if not engine.is_trained:
        engine.train_models()
    return engine.results


@app.get("/api/eda")
def get_eda():
    if engine.df is None:
        engine.load_and_preprocess()
    return engine.results.get("eda_summary", engine._get_eda_summary())


@app.post("/api/predict")
def predict_scenario(req: PredictRequest):
    try:
        pred = engine.predict_scenario(
            demand=req.demand,
            usd_index=req.usd_index,
            opec_policy=req.opec_policy,
            lag1=req.lag1,
            lag2=req.lag2,
            lag3=req.lag3,
            lag6=req.lag6,
            month=req.month or 10,
            year=req.year or 2024
        )
        return pred
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/retrain")
def retrain(test_split: float = 0.30, tune_iter: int = 15):
    try:
        res = engine.train_models(test_split=test_split, tune_iter=tune_iter)
        return {"status": "success", "results": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not (file.filename.endswith('.xlsx') or file.filename.endswith('.xls') or file.filename.endswith('.csv')):
        raise HTTPException(status_code=400, detail="Only Excel (.xlsx/.xls) or CSV files are supported.")

    file_path = os.path.join(BASE_DIR, "Oil prices 2.xlsx" if file.filename.endswith('.xlsx') else "uploaded_data.csv")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        engine.data_path = file_path
        engine.load_and_preprocess(file_path)
        res = engine.train_models()
        return {"status": "success", "filename": file.filename, "records": len(engine.df), "results": res}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process file: {str(e)}")


@app.get("/api/export")
def export_predictions():
    export_path = os.path.join(BASE_DIR, "final_predictions_improved.xlsx")
    if not os.path.exists(export_path):
        if engine.df is not None:
            engine.df.to_excel(export_path, index=False)
        else:
            raise HTTPException(status_code=404, detail="No predictions file available.")
    return FileResponse(
        export_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="final_predictions_improved.xlsx"
    )


@app.get("/api/data")
def get_data(page: int = 1, page_size: int = 50, search: str = ""):
    if engine.df is None:
        if not engine.is_trained:
            engine.train_models()
        else:
            engine.load_and_preprocess()

    df = engine.df.copy()
    if search:
        search_lower = search.lower()
        mask = df.astype(str).apply(lambda row: row.str.lower().str.contains(search_lower).any(), axis=1)
        df = df[mask]

    total_rows = len(df)
    start = (page - 1) * page_size
    end = start + page_size
    page_df = df.iloc[start:end]

    cols_order = [
        'Date', 'Price (USD/bbl)', 'SVR_Prediction', 'RF_Prediction', 'Naive_Prediction',
        'SVR_Residual', 'RF_Residual', 'Demand (Mbbl/day)', 'USD Index', 'OPEC Policy',
        'Price_Lag1', 'Price_MoM_Change', 'Rolling_Mean_3', 'Rolling_Std_5', 'Is_Outlier'
    ]
    available_cols = [c for c in cols_order if c in page_df.columns]
    
    # Format rows
    records = []
    for _, row in page_df[available_cols].iterrows():
        rec = {}
        for c in available_cols:
            val = row[c]
            if pd.isna(val):
                rec[c] = "-"
            elif isinstance(val, pd.Timestamp):
                rec[c] = val.strftime('%Y-%m-%d')
            elif isinstance(val, (float, np.floating)):
                rec[c] = round(float(val), 2)
            elif isinstance(val, (bool, np.bool_)):
                rec[c] = bool(val)
            else:
                rec[c] = val
        records.append(rec)

    return {
        "total": total_rows,
        "page": page,
        "page_size": page_size,
        "columns": available_cols,
        "records": records
    }
