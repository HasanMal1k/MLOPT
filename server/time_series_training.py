from fastapi import APIRouter, HTTPException, BackgroundTasks, Form, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from io import StringIO, BytesIO
import pandas as pd
import numpy as np
import json
import uuid
import os
import logging
import tempfile
import shutil
from datetime import datetime
import warnings

# Time series specific imports
try:
    from darts import TimeSeries
    from darts.models import (
        NaiveSeasonal, ARIMA, ExponentialSmoothing, NBEATSModel, RNNModel,
        VARIMA, LinearRegressionModel, Prophet, AutoARIMA, Theta,
        FFT, KalmanForecaster, RandomForest, LightGBMModel
    )
    from darts.metrics import smape, mae, rmse, mape, ope
    DARTS_AVAILABLE = True
except ImportError:
    DARTS_AVAILABLE = False
    logger.warning("Darts library not available. Time series training will be limited.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/time-series", tags=["Time Series Training"])

# Global storage for time series training tasks
ts_training_tasks: Dict[str, Dict] = {}

# Pydantic models
class TimeSeriesConfigRequest(BaseModel):
    filename: str
    forecasting_type: str  # "univariate", "multivariate", "exogenous"
    target_column: str
    date_column: str
    exogenous_columns: Optional[List[str]] = None
    forecast_horizon: int = 12
    train_split: float = 0.8
    seasonal_periods: Optional[int] = None
    include_deep_learning: bool = True
    include_statistical: bool = True
    include_ml: bool = True
    max_epochs: int = 10
    session_id: int = 123

class TimeSeriesAnalysisRequest(BaseModel):
    filename: str
    date_column: str
    value_columns: List[str]

# Helper functions
def detect_time_series_patterns(series: pd.Series, freq: str = None) -> Dict:
    """Analyze time series patterns and characteristics"""
    try:
        patterns = {
            "length": len(series),
            "missing_values": series.isna().sum(),
            "missing_percentage": (series.isna().sum() / len(series)) * 100,
            "min_value": float(series.min()) if not series.isna().all() else None,
            "max_value": float(series.max()) if not series.isna().all() else None,
            "mean_value": float(series.mean()) if not series.isna().all() else None,
            "std_value": float(series.std()) if not series.isna().all() else None,
            "trend": "unknown",
            "seasonality": "unknown",
            "stationarity": "unknown"
        }
        
        # Basic trend analysis (simple linear regression slope)
        if len(series.dropna()) > 1:
            x = np.arange(len(series))
            valid_idx = ~series.isna()
            if valid_idx.sum() > 1:
                slope = np.polyfit(x[valid_idx], series[valid_idx], 1)[0]
                if abs(slope) < 0.01:
                    patterns["trend"] = "no_trend"
                elif slope > 0:
                    patterns["trend"] = "increasing"
                else:
                    patterns["trend"] = "decreasing"
        
        # Basic seasonality detection (simplified)
        if len(series.dropna()) > 24:  # Need enough data
            try:
                # Check for weekly seasonality (7 periods)
                weekly_corr = series.autocorr(lag=7) if len(series) > 7 else 0
                # Check for monthly seasonality (30 periods)
                monthly_corr = series.autocorr(lag=30) if len(series) > 30 else 0
                
                if abs(weekly_corr) > 0.3 or abs(monthly_corr) > 0.3:
                    patterns["seasonality"] = "likely_seasonal"
                else:
                    patterns["seasonality"] = "likely_non_seasonal"
            except:
                patterns["seasonality"] = "unknown"
        
        return patterns
        
    except Exception as e:
        logger.warning(f"Error analyzing time series patterns: {e}")
        return {
            "length": len(series) if series is not None else 0,
            "missing_values": 0,
            "missing_percentage": 0,
            "min_value": None,
            "max_value": None,
            "mean_value": None,
            "std_value": None,
            "trend": "unknown",
            "seasonality": "unknown",
            "stationarity": "unknown"
        }

def prepare_darts_series(df: pd.DataFrame, date_col: str, value_cols: List[str]) -> Dict:
    """Convert pandas DataFrame to Darts TimeSeries objects"""
    try:
        if not DARTS_AVAILABLE:
            raise Exception("Darts library not available")
        
        # Ensure date column is datetime
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        
        # Set date as index
        df_indexed = df.set_index(date_col)
        
        series_dict = {}
        
        if len(value_cols) == 1:
            # Univariate case
            series_dict['main'] = TimeSeries.from_series(df_indexed[value_cols[0]])
        else:
            # Multivariate case
            for col in value_cols:
                if col in df_indexed.columns:
                    series_dict[col] = TimeSeries.from_series(df_indexed[col])
        
        return {
            "success": True,
            "series": series_dict,
            "frequency": None,  # Auto-detect or infer
            "message": f"Created {len(series_dict)} time series"
        }
        
    except Exception as e:
        logger.error(f"Error preparing Darts series: {e}")
        return {
            "success": False,
            "series": {},
            "frequency": None,
            "message": str(e)
        }

async def run_time_series_training(config_id: str, config: dict):
    """Background task for time series model training"""
    try:
        if not DARTS_AVAILABLE:
            ts_training_tasks[config_id]["status"] = "failed"
            ts_training_tasks[config_id]["error"] = "Darts library not available. Please install with: pip install darts"
            return
        
        # Update status
        ts_training_tasks[config_id]["status"] = "loading_data"
        
        # Load saved training data
        data_file = ts_training_tasks[config_id]["data_file"]
        df = pd.read_csv(data_file)
        
        logger.info(f"Time series training data shape: {df.shape}")
        
        ts_training_tasks[config_id]["status"] = "preparing_series"
        
        # Prepare time series data
        forecasting_type = config["forecasting_type"]
        date_column = config["date_column"]
        target_column = config["target_column"]
        exogenous_columns = config.get("exogenous_columns", [])
        forecast_horizon = config.get("forecast_horizon", 12)
        train_split = config.get("train_split", 0.8)
        max_epochs = config.get("max_epochs", 10)
        
        # Convert to datetime and sort
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column).reset_index(drop=True)
        df_indexed = df.set_index(date_column)
        
        # Prepare series based on forecasting type
        if forecasting_type == "univariate":
            target_series = TimeSeries.from_series(df_indexed[target_column])
            train_target, val_target = target_series.split_after(train_split)
            
            series_info = {
                "type": "univariate",
                "target_series": target_series,
                "train_target": train_target,
                "val_target": val_target,
                "exo_series": None,
                "train_exo": None,
                "val_exo": None
            }
            
        elif forecasting_type == "multivariate":
            # Create multiple series for each column
            value_columns = [col for col in df.columns if col != date_column and col in df.select_dtypes(include=[np.number]).columns]
            
            series_dict = {}
            for col in value_columns:
                series_dict[col] = TimeSeries.from_series(df_indexed[col])
            
            series_info = {
                "type": "multivariate",
                "series_dict": series_dict,
                "value_columns": value_columns
            }
            
        elif forecasting_type == "exogenous":
            target_series = TimeSeries.from_series(df_indexed[target_column])
            
            if exogenous_columns and len(exogenous_columns) > 0:
                exo_series = TimeSeries.from_dataframe(df_indexed, value_cols=exogenous_columns)
            else:
                # Auto-select numeric columns as exogenous
                numeric_cols = [col for col in df.columns 
                              if col != date_column and col != target_column 
                              and col in df.select_dtypes(include=[np.number]).columns]
                if numeric_cols:
                    exo_series = TimeSeries.from_dataframe(df_indexed, value_cols=numeric_cols)
                else:
                    exo_series = None
            
            train_target, val_target = target_series.split_after(train_split)
            
            if exo_series is not None:
                train_exo, val_exo = exo_series.split_after(train_split)
            else:
                train_exo, val_exo = None, None
            
            series_info = {
                "type": "exogenous",
                "target_series": target_series,
                "train_target": train_target,
                "val_target": val_target,
                "exo_series": exo_series,
                "train_exo": train_exo,
                "val_exo": val_exo
            }
        
        ts_training_tasks[config_id]["status"] = "training_models"
        logger.info(f"Starting time series model training for {forecasting_type}")
        
        # Define model candidates based on configuration
        results = []
        
        if forecasting_type == "univariate":
            results = await train_univariate_models(series_info, config, max_epochs)
        elif forecasting_type == "multivariate":
            results = await train_multivariate_models(series_info, config, max_epochs)
        elif forecasting_type == "exogenous":
            results = await train_exogenous_models(series_info, config, max_epochs)
        
        ts_training_tasks[config_id]["status"] = "saving_models"
        
        # Create models directory
        models_dir = Path("models") / config_id
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Save successful models
        saved_models = []
        for i, result in enumerate(results):
            if result["status"] == "ok" and "model_object" in result:
                try:
                    model_path = models_dir / f"{result['model']}_{i}.pkl"
                    result["model_object"].save(str(model_path))
                    saved_models.append({
                        "name": result["model"],
                        "path": str(model_path),
                        "metrics": {k: v for k, v in result.items() if k not in ["model_object", "status"]}
                    })
                except Exception as e:
                    logger.warning(f"Failed to save model {result['model']}: {e}")
        
        # Create leaderboard
        leaderboard_df = pd.DataFrame([
            {k: v for k, v in result.items() if k != "model_object"} 
            for result in results
        ])
        
        # Save leaderboard
        leaderboard_df.to_csv(models_dir / 'leaderboard.csv', index=False)
        
        # Get best model (lowest SMAPE or MAE)
        successful_results = [r for r in results if r["status"] == "ok"]
        if successful_results:
            # Sort by SMAPE first, then MAE
            best_model = min(successful_results, 
                           key=lambda x: (x.get("smape", float('inf')) or float('inf')))
        else:
            best_model = None
        
        # Update final status
        ts_training_tasks[config_id].update({
            "status": "completed",
            "leaderboard": leaderboard_df.to_dict('records'),
            "best_model_name": best_model["model"] if best_model else "None",
            "models_saved": len(saved_models),
            "models_dir": str(models_dir),
            "completed_at": datetime.now().isoformat(),
            "total_models_tested": len(results),
            "successful_models": len(successful_results),
            "forecasting_type": forecasting_type,
            "forecast_horizon": forecast_horizon
        })
        
        logger.info(f"Time series training completed for {config_id}. Tested {len(results)} models, {len(successful_results)} successful.")
        
        # Clean up training data file
        try:
            os.unlink(data_file)
        except:
            pass
        
    except Exception as e:
        logger.error(f"Time series training error for {config_id}: {str(e)}")
        ts_training_tasks[config_id]["status"] = "failed"
        ts_training_tasks[config_id]["error"] = str(e)

async def train_univariate_models(series_info: Dict, config: Dict, max_epochs: int) -> List[Dict]:
    """Train univariate forecasting models"""
    results = []
    train_target = series_info["train_target"]
    val_target = series_info["val_target"]
    
    # Define model candidates
    candidates = {}
    
    # Statistical models
    if config.get("include_statistical", True):
        candidates.update({
            "NaiveSeasonal": NaiveSeasonal(K=config.get("seasonal_periods", 12)),
            "Theta": Theta(),
        })
        
        # Add AutoARIMA (more robust than ARIMA)
        try:
            candidates["AutoARIMA"] = AutoARIMA()
        except Exception as e:
            logger.warning(f"AutoARIMA not available: {e}")
        
        # Add ExponentialSmoothing
        try:
            candidates["ExponentialSmoothing"] = ExponentialSmoothing()
        except Exception as e:
            logger.warning(f"ExponentialSmoothing not available: {e}")
        
        # Add Prophet if available
        try:
            from darts.models import Prophet
            candidates["Prophet"] = Prophet()
        except Exception as e:
            logger.warning(f"Prophet not available: {e}")
    
    # Machine Learning models
    if config.get("include_ml", True):
        try:
            # More robust configuration for ML models
            lags = min(24, len(train_target) // 3)  # Use up to 24 lags
            
            candidates.update({
                "RandomForest": RandomForest(
                    lags=lags,
                    output_chunk_length=min(12, len(val_target)),
                    n_estimators=100,  # More trees
                    random_state=42
                ),
                "LightGBM": LightGBMModel(
                    lags=lags,
                    output_chunk_length=min(12, len(val_target)),
                    random_state=42,
                    n_estimators=100
                ),
            })
        except Exception as e:
            logger.warning(f"ML models not available: {e}")
    
    # Deep Learning models - FIXED THRESHOLDS
    if config.get("include_deep_learning", True):
        # Calculate proper chunk lengths
        min_data_points = 100  # Minimum for deep learning
        
        if len(train_target) >= min_data_points:
            try:
                # Calculate appropriate chunk lengths
                input_chunk = max(12, min(24, len(train_target) // 5))
                output_chunk = max(6, min(12, len(val_target)))
                
                # Increase epochs for better learning
                dl_epochs = max(max_epochs, 20)  # At least 20 epochs
                
                candidates.update({
                    "NBEATS": NBEATSModel(
                        input_chunk_length=input_chunk,
                        output_chunk_length=output_chunk,
                        n_epochs=dl_epochs,
                        num_stacks=10,  # Default stacks
                        num_blocks=1,
                        num_layers=2,
                        layer_widths=256,
                        random_state=42,
                        pl_trainer_kwargs={
                            "enable_progress_bar": False,
                            "enable_model_summary": False
                        }
                    ),
                    "LSTM": RNNModel(
                        model="LSTM",
                        hidden_dim=25,
                        n_rnn_layers=1,
                        input_chunk_length=input_chunk,
                        training_length=input_chunk + output_chunk,
                        n_epochs=dl_epochs,
                        dropout=0.1,
                        random_state=42,
                        pl_trainer_kwargs={
                            "enable_progress_bar": False,
                            "enable_model_summary": False
                        }
                    ),
                    "GRU": RNNModel(
                        model="GRU",
                        hidden_dim=25,
                        n_rnn_layers=1,
                        input_chunk_length=input_chunk,
                        training_length=input_chunk + output_chunk,
                        n_epochs=dl_epochs,
                        dropout=0.1,
                        random_state=42,
                        pl_trainer_kwargs={
                            "enable_progress_bar": False,
                            "enable_model_summary": False
                        }
                    )
                })
                
                logger.info(f"Deep learning enabled: input_chunk={input_chunk}, output_chunk={output_chunk}, epochs={dl_epochs}")
                
            except Exception as e:
                logger.warning(f"Deep learning models not available: {e}")
        else:
            logger.info(f"Skipping deep learning: need {min_data_points}+ points, have {len(train_target)}")
    
    # Train each model
    for name, model in candidates.items():
        try:
            logger.info(f"Training univariate model: {name}")
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(train_target)
                pred = model.predict(len(val_target))
            
            # Calculate metrics
            metrics = {
                "model": name,
                "status": "ok",
                "smape": float(smape(val_target, pred)),
                "mae": float(mae(val_target, pred)),
                "rmse": float(rmse(val_target, pred)),
                "mape": float(mape(val_target, pred)) if len(val_target) > 0 else None,
                "error": None,
                "model_object": model
            }
            
            results.append(metrics)
            logger.info(f"✓ {name}: SMAPE={metrics['smape']:.4f}, MAE={metrics['mae']:.4f}")
            
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"✗ {name} failed: {error_msg}")
            results.append({
                "model": name,
                "status": "failed",
                "smape": None,
                "mae": None,
                "rmse": None,
                "mape": None,
                "error": error_msg
            })
    
    return results

async def train_multivariate_models(series_info: Dict, config: Dict, max_epochs: int) -> List[Dict]:
    """Train multivariate forecasting models"""
    results = []
    series_dict = series_info["series_dict"]
    value_columns = series_info["value_columns"]
    
    # Train models for each series
    for series_name, series in series_dict.items():
        try:
            train, val = series.split_after(config.get("train_split", 0.8))
            
            candidates = {}
            
            # Statistical models
            if config.get("include_statistical", True) and len(value_columns) > 1:
                try:
                    # VARIMA needs multivariate input
                    candidates["VARIMA"] = VARIMA()
                except Exception as e:
                    logger.warning(f"VARIMA not available for {series_name}: {e}")
            
            # Add univariate statistical models
            candidates.update({
                "NaiveSeasonal": NaiveSeasonal(K=config.get("seasonal_periods", 12)),
                "Theta": Theta(),
            })
            
            # Machine Learning
            if config.get("include_ml", True):
                try:
                    lags = min(24, len(train) // 3)
                    candidates.update({
                        "RandomForest": RandomForest(
                            lags=lags,
                            output_chunk_length=min(12, len(val)),
                            n_estimators=100,
                            random_state=42
                        ),
                        "LightGBM": LightGBMModel(
                            lags=lags,
                            output_chunk_length=min(12, len(val)),
                            n_estimators=100,
                            random_state=42
                        )
                    })
                except Exception as e:
                    logger.warning(f"ML models not available for {series_name}: {e}")
            
            # Deep Learning models
            if config.get("include_deep_learning", True) and len(train) >= 100:
                try:
                    input_chunk = max(12, min(24, len(train) // 5))
                    output_chunk = max(6, min(12, len(val)))
                    dl_epochs = max(max_epochs, 20)
                    
                    candidates.update({
                        "NBEATS": NBEATSModel(
                            input_chunk_length=input_chunk,
                            output_chunk_length=output_chunk,
                            n_epochs=dl_epochs,
                            num_stacks=10,
                            num_blocks=1,
                            num_layers=2,
                            layer_widths=256,
                            random_state=42,
                            pl_trainer_kwargs={
                                "enable_progress_bar": False,
                                "enable_model_summary": False
                            }
                        ),
                        "LSTM": RNNModel(
                            model="LSTM",
                            hidden_dim=25,
                            n_rnn_layers=1,
                            input_chunk_length=input_chunk,
                            training_length=input_chunk + output_chunk,
                            n_epochs=dl_epochs,
                            dropout=0.1,
                            random_state=42,
                            pl_trainer_kwargs={
                                "enable_progress_bar": False,
                                "enable_model_summary": False
                            }
                        )
                    })
                except Exception as e:
                    logger.warning(f"Deep learning models not available for {series_name}: {e}")
            
            # Train each model for this series
            for model_name, model in candidates.items():
                try:
                    logger.info(f"Training {model_name} for series {series_name}")
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model.fit(train)
                        pred = model.predict(len(val))
                    
                    metrics = {
                        "series": series_name,
                        "model": model_name,
                        "status": "ok",
                        "smape": float(smape(val, pred)),
                        "mae": float(mae(val, pred)),
                        "rmse": float(rmse(val, pred)),
                        "mape": float(mape(val, pred)) if len(val) > 0 else None,
                        "error": None,
                        "model_object": model
                    }
                    
                    results.append(metrics)
                    logger.info(f"✓ {series_name}-{model_name}: SMAPE={metrics['smape']:.4f}")
                    
                except Exception as e:
                    error_msg = str(e)
                    logger.warning(f"✗ {series_name}-{model_name} failed: {error_msg}")
                    results.append({
                        "series": series_name,
                        "model": model_name,
                        "status": "failed",
                        "smape": None,
                        "mae": None,
                        "rmse": None,
                        "mape": None,
                        "error": error_msg
                    })
                    
        except Exception as e:
            logger.error(f"Error processing series {series_name}: {e}")
    
    return results

async def train_exogenous_models(series_info: Dict, config: Dict, max_epochs: int) -> List[Dict]:
    """Train exogenous forecasting models"""
    results = []
    train_target = series_info["train_target"]
    val_target = series_info["val_target"]
    train_exo = series_info["train_exo"]
    val_exo = series_info["val_exo"]
    
    # Define model candidates
    candidates = {}
    
    # Calculate appropriate lags
    lags = min(24, len(train_target) // 3)
    lags_exo = [0, 1, 2, 3, 6, 12] if train_exo is not None else None
    
    # Statistical/ML models with exogenous support
    if config.get("include_statistical", True):
        try:
            candidates["LinearRegression"] = LinearRegressionModel(
                lags=lags,
                lags_past_covariates=lags_exo,
                output_chunk_length=min(12, len(val_target))
            )
        except Exception as e:
            logger.warning(f"LinearRegression not available: {e}")
    
    # Machine Learning models
    if config.get("include_ml", True):
        try:
            candidates.update({
                "RandomForest_Exo": RandomForest(
                    lags=lags,
                    lags_past_covariates=lags_exo if train_exo is not None else None,
                    output_chunk_length=min(12, len(val_target)),
                    n_estimators=100,
                    random_state=42
                ),
                "LightGBM_Exo": LightGBMModel(
                    lags=lags,
                    lags_past_covariates=lags_exo if train_exo is not None else None,
                    output_chunk_length=min(12, len(val_target)),
                    n_estimators=100,
                    random_state=42
                )
            })
        except Exception as e:
            logger.warning(f"ML exogenous models not available: {e}")
    
    # Deep Learning models
    if config.get("include_deep_learning", True) and len(train_target) >= 100:
        try:
            input_chunk = max(12, min(24, len(train_target) // 5))
            output_chunk = max(6, min(12, len(val_target)))
            dl_epochs = max(max_epochs, 20)
            
            candidates.update({
                "NBEATS_Exo": NBEATSModel(
                    input_chunk_length=input_chunk,
                    output_chunk_length=output_chunk,
                    n_epochs=dl_epochs,
                    num_stacks=10,
                    num_blocks=1,
                    num_layers=2,
                    layer_widths=256,
                    random_state=42,
                    pl_trainer_kwargs={
                        "enable_progress_bar": False,
                        "enable_model_summary": False
                    }
                ),
                "LSTM_Exo": RNNModel(
                    model="LSTM",
                    hidden_dim=25,
                    n_rnn_layers=1,
                    input_chunk_length=input_chunk,
                    training_length=input_chunk + output_chunk,
                    n_epochs=dl_epochs,
                    dropout=0.1,
                    random_state=42,
                    pl_trainer_kwargs={
                        "enable_progress_bar": False,
                        "enable_model_summary": False
                    }
                )
            })
        except Exception as e:
            logger.warning(f"Deep learning exogenous models not available: {e}")
    
    # Train each model
    for name, model in candidates.items():
        try:
            logger.info(f"Training exogenous model: {name}")
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # FIX: All models should use past_covariates when available
                if train_exo is not None:
                    model.fit(train_target, past_covariates=train_exo)
                    pred = model.predict(len(val_target), past_covariates=val_exo)
                else:
                    # Fallback to univariate if no exogenous variables
                    model.fit(train_target)
                    pred = model.predict(len(val_target))
            
            metrics = {
                "model": name,
                "status": "ok",
                "smape": float(smape(val_target, pred)),
                "mae": float(mae(val_target, pred)),
                "rmse": float(rmse(val_target, pred)),
                "mape": float(mape(val_target, pred)) if len(val_target) > 0 else None,
                "exogenous_used": train_exo is not None,
                "error": None,
                "model_object": model
            }
            
            results.append(metrics)
            logger.info(f"✓ {name}: SMAPE={metrics['smape']:.4f}, MAE={metrics['mae']:.4f}")
            
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"✗ {name} failed: {error_msg}")
            results.append({
                "model": name,
                "status": "failed",
                "smape": None,
                "mae": None,
                "rmse": None,
                "mape": None,
                "exogenous_used": train_exo is not None,
                "error": error_msg
            })
    
    return results

# API Endpoints

@router.post("/analyze-time-series-with-file/")
async def analyze_time_series_with_file(
    file: UploadFile = File(...),
    date_column: str = Form(...),
    value_columns: str = Form(...)  # JSON string of column names
):
    """Analyze time series data from uploaded file"""
    try:
        # Parse value columns
        try:
            value_cols = json.loads(value_columns)
        except:
            value_cols = [value_columns]  # Single column case
        
        # Read the uploaded file
        contents = await file.read()
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(StringIO(contents.decode('utf-8')))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Validate columns exist
        missing_cols = [col for col in [date_column] + value_cols if col not in df.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Columns not found: {missing_cols}")
        
        # Convert date column
        try:
            df[date_column] = pd.to_datetime(df[date_column])
        except:
            raise HTTPException(status_code=400, detail=f"Cannot convert {date_column} to datetime")
        
        # Sort by date
        df = df.sort_values(date_column)
        
        # Analyze each value column
        analysis_results = {}
        
        for col in value_cols:
            if col in df.columns:
                # Ensure numeric
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass
                
                patterns = detect_time_series_patterns(df[col])
                analysis_results[col] = patterns
        
        # Overall dataset info
        dataset_info = {
            "total_rows": len(df),
            "date_range": {
                "start": df[date_column].min().isoformat() if not df[date_column].isna().all() else None,
                "end": df[date_column].max().isoformat() if not df[date_column].isna().all() else None
            },
            "frequency": "unknown",  # Could be inferred
            "missing_dates": 0,  # Could be calculated
            "duplicate_dates": df[date_column].duplicated().sum()
        }
        
        # Recommendations
        recommendations = {
            "forecasting_type": "univariate" if len(value_cols) == 1 else "multivariate",
            "suitable_for_deep_learning": len(df) > 100,
            "seasonal_periods": 12,  # Default assumption
            "forecast_horizon": min(12, len(df) // 10),
            "suggested_models": []
        }
        
        # Model recommendations based on data characteristics
        if len(df) > 100:
            recommendations["suggested_models"].extend(["NBEATS", "LSTM", "AutoARIMA"])
        else:
            recommendations["suggested_models"].extend(["ExponentialSmoothing", "NaiveSeasonal"])
        
        if len(value_cols) > 1:
            recommendations["suggested_models"].append("VARIMA")
        
        return JSONResponse({
            "success": True,
            "dataset_info": dataset_info,
            "column_analysis": analysis_results,
            "recommendations": recommendations,
            "darts_available": DARTS_AVAILABLE
        })
        
    except Exception as e:
        logger.error(f"Time series analysis error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/configure-time-series-training/")
async def configure_time_series_training(
    file: UploadFile = File(...),
    forecasting_type: str = Form(...),
    target_column: str = Form(...),
    date_column: str = Form(...),
    exogenous_columns: str = Form("[]"),
    forecast_horizon: int = Form(12),
    train_split: float = Form(0.8),
    seasonal_periods: int = Form(12),
    include_deep_learning: bool = Form(True),
    include_statistical: bool = Form(True),
    include_ml: bool = Form(True),
    max_epochs: int = Form(10)
):
    """Configure time series training with uploaded file"""
    try:
        if not DARTS_AVAILABLE:
            raise HTTPException(status_code=400, detail="Darts library not available. Please install with: pip install darts")
        
        # Parse exogenous columns
        try:
            exogenous_cols = json.loads(exogenous_columns)
        except:
            exogenous_cols = []
        
        # Read the uploaded file
        contents = await file.read()
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(StringIO(contents.decode('utf-8')))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Validate required columns
        required_cols = [date_column, target_column]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Required columns not found: {missing_cols}")
        
        # Validate exogenous columns if provided
        if exogenous_cols:
            missing_exo = [col for col in exogenous_cols if col not in df.columns]
            if missing_exo:
                raise HTTPException(status_code=400, detail=f"Exogenous columns not found: {missing_exo}")
        
        # Validate parameters
        if not 0.1 <= train_split <= 0.9:
            raise HTTPException(status_code=400, detail="Train split must be between 0.1 and 0.9")
        
        if forecast_horizon < 1:
            raise HTTPException(status_code=400, detail="Forecast horizon must be at least 1")
        
        # Convert date column and sort
        try:
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.sort_values(date_column).reset_index(drop=True)
        except:
            raise HTTPException(status_code=400, detail=f"Cannot convert {date_column} to datetime")
        
        # Ensure target is numeric
        try:
            df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
        except:
            raise HTTPException(status_code=400, detail=f"Target column {target_column} must be numeric")
        
        # Check for sufficient data
        if len(df) < 20:
            raise HTTPException(status_code=400, detail="Need at least 20 data points for time series forecasting")
        
        # Generate unique config ID
        config_id = str(uuid.uuid4())
        
        # Save dataset for training
        training_data_dir = Path("training_data")
        training_data_dir.mkdir(exist_ok=True)
        training_file_path = training_data_dir / f"{config_id}_ts_data.csv"
        df.to_csv(training_file_path, index=False)
        
        # Store configuration for training
        ts_training_tasks[config_id] = {
            "status": "configured",
            "config": {
                "forecasting_type": forecasting_type,
                "target_column": target_column,
                "date_column": date_column,
                "exogenous_columns": exogenous_cols,
                "forecast_horizon": forecast_horizon,
                "train_split": train_split,
                "seasonal_periods": seasonal_periods,
                "include_deep_learning": include_deep_learning,
                "include_statistical": include_statistical,
                "include_ml": include_ml,
                "max_epochs": max_epochs
            },
            "data_file": str(training_file_path),
            "data_shape": df.shape,
            "date_range": {
                "start": df[date_column].min().isoformat(),
                "end": df[date_column].max().isoformat()
            },
            "original_filename": file.filename
        }
        
        logger.info(f"Time series configuration saved with ID: {config_id}")
        
        return JSONResponse({
            "success": True,
            "config_id": config_id,
            "data_shape": df.shape,
            "date_range": ts_training_tasks[config_id]["date_range"],
            "forecasting_type": forecasting_type,
            "message": "Time series training configuration validated successfully"
        })
        
    except Exception as e:
        logger.error(f"Time series configuration error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/start-time-series-training/")
async def start_time_series_training(background_tasks: BackgroundTasks, config_id: str = Form(...)):
    """Start time series model training in background"""
    try:
        if config_id not in ts_training_tasks:
            raise HTTPException(status_code=404, detail="Configuration not found")
        
        if not DARTS_AVAILABLE:
            raise HTTPException(status_code=400, detail="Darts library not available")
        
        # Get config
        config = ts_training_tasks[config_id]["config"]
        
        # Add timestamp
        ts_training_tasks[config_id]["started_at"] = datetime.now().isoformat()
        
        # Start background training
        background_tasks.add_task(run_time_series_training, config_id, config)
        
        ts_training_tasks[config_id]["status"] = "started"
        
        return JSONResponse({
            "success": True,
            "task_id": config_id,
            "status": "started",
            "message": "Time series training started in background",
            "started_at": ts_training_tasks[config_id]["started_at"]
        })
        
    except Exception as e:
        logger.error(f"Start time series training error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/time-series-training-status/{task_id}")
async def get_time_series_training_status(task_id: str):
    """Get time series training status and results"""
    if task_id not in ts_training_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = ts_training_tasks[task_id]
    
    response = {
        "task_id": task_id,
        "status": task["status"],
        "forecasting_type": task.get("config", {}).get("forecasting_type"),
        "data_shape": task.get("data_shape"),
        "date_range": task.get("date_range")
    }
    
    if task["status"] == "completed":
        response.update({
            "leaderboard": task["leaderboard"],
            "best_model_name": task["best_model_name"],
            "models_saved": task["models_saved"],
            "total_models_tested": task["total_models_tested"],
            "successful_models": task["successful_models"],
            "forecast_horizon": task["forecast_horizon"]
        })
    elif task["status"] == "failed":
        response["error"] = task.get("error")
    
    return JSONResponse(response)

@router.get("/download-ts-model/{task_id}/{model_name}")
async def download_time_series_model(task_id: str, model_name: str):
    """Download trained time series model"""
    if task_id not in ts_training_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = ts_training_tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Training not completed")
    
    models_dir = Path("models") / task_id
    
    # Look for the model file
    model_files = list(models_dir.glob(f"{model_name}*.pkl"))
    if not model_files:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_path = model_files[0]
    
    return FileResponse(
        path=model_path,
        filename=f"ts_{model_name}_{task_id}.pkl",
        media_type="application/octet-stream"
    )

@router.get("/download-ts-leaderboard/{task_id}")
async def download_time_series_leaderboard(task_id: str):
    """Download time series leaderboard CSV"""
    if task_id not in ts_training_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = ts_training_tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Training not completed")
    
    leaderboard_path = Path("models") / task_id / "leaderboard.csv"
    if not leaderboard_path.exists():
        raise HTTPException(status_code=404, detail="Leaderboard not found")
    
    return FileResponse(
        path=leaderboard_path,
        filename=f"ts_leaderboard_{task_id}.csv",
        media_type="text/csv"
    )

@router.get("/available-ts-models/")
async def get_available_time_series_models():
    """Get list of available time series models"""
    models_info = {
        "darts_available": DARTS_AVAILABLE,
        "statistical_models": [
            "NaiveSeasonal", "AutoARIMA", "ExponentialSmoothing", 
            "Theta", "Prophet", "VARIMA"
        ],
        "machine_learning_models": [
            "RandomForest", "LightGBM", "LinearRegression"
        ],
        "deep_learning_models": [
            "NBEATS", "LSTM", "GRU"
        ],
        "forecasting_types": [
            "univariate", "multivariate", "exogenous"
        ]
    }
    
    if not DARTS_AVAILABLE:
        models_info["error"] = "Darts library not available. Install with: pip install darts"
    
    return JSONResponse(models_info)

@router.delete("/clear-ts-cache/")
async def clear_time_series_cache():
    """Clear time series training tasks cache"""
    global ts_training_tasks
    ts_training_tasks.clear()
    
    return JSONResponse({
        "success": True,
        "message": "Time series cache cleared successfully"
    })