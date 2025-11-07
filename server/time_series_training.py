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
"""
Enhanced Time Series Training Module with 40+ Models

STATISTICAL MODELS (CPU Efficient, <1GB RAM):
- Naive models: NaiveSeasonal, NaiveDrift
- ARIMA family: AutoARIMA, ARIMA(1,0,1), ARIMA(1,1,1), ARIMA(2,1,2)
- Exponential Smoothing: Basic, Additive, Multiplicative, Holt, Holt-Winters
- Advanced: Theta, Prophet (+ variations), FFT, Kalman Filter, Croston
- StatsForecast: AutoARIMA, AutoETS, AutoCES

MACHINE LEARNING MODELS (1-4GB RAM):
- Tree-based: RandomForest (full & light), LightGBM (full & fast), XGBoost, CatBoost
- Linear: LinearRegression, Ridge, Lasso, ElasticNet

DEEP LEARNING MODELS (CPU-Only, 2-8GB RAM):
- Neural Networks: NBEATS (tiny & medium), LSTM/GRU (small & medium)
- Advanced: TCN (Temporal CNN), Transformer (small)
- All optimized for CPU training with minimal hardware requirements

FORECASTING TYPES:
- Univariate: Single time series forecasting
- Multivariate: Multiple related time series
- Exogenous: Using external predictors/covariates
"""

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
        
        # Ensure date column is datetime with robust parsing
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce', format='mixed')
        # Remove rows with invalid dates
        df = df.dropna(subset=[date_col])
        df = df.sort_values(date_col)
        
        # Check for duplicate timestamps and aggregate them
        duplicate_count = df[date_col].duplicated().sum()
        if duplicate_count > 0:
            logger.warning(f"Found {duplicate_count} duplicate timestamps in analysis. Aggregating by mean...")
            # Group by date column and aggregate numeric columns by mean
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Create aggregation dict - keep all value columns
            agg_dict = {col: 'mean' for col in value_cols if col in df.columns}
            
            # Group and aggregate
            df = df.groupby(date_col, as_index=False).agg(agg_dict)
            logger.info(f"After aggregation: {len(df)} unique timestamps")
        
        # Set date as index
        df_indexed = df.set_index(date_col)
        
        # Detect frequency
        def detect_freq_helper(df_with_date_index):
            try:
                inferred_freq = pd.infer_freq(df_with_date_index.index)
                if inferred_freq:
                    return inferred_freq
                
                time_diffs = df_with_date_index.index.to_series().diff().dropna()
                most_common_diff = time_diffs.mode()
                
                if len(most_common_diff) > 0:
                    diff_days = most_common_diff.iloc[0].days
                    if diff_days == 1:
                        weekdays = df_with_date_index.index.weekday
                        has_weekends = any(day >= 5 for day in weekdays)
                        return 'D' if has_weekends else 'B'
                    elif diff_days == 7:
                        return 'W'
                    elif diff_days >= 28 and diff_days <= 31:
                        return 'M'
                return 'B'
            except:
                return 'B'
        
        detected_freq = detect_freq_helper(df_indexed)
        
        # Safe creation helper (same as in main function)
        def safe_create_ts(data, column_name, freq):
            try:
                if freq is not None:
                    return TimeSeries.from_series(
                        data[column_name] if isinstance(data, pd.DataFrame) else data,
                        fill_missing_dates=True,
                        freq=freq
                    )
            except Exception as e:
                logger.warning(f"Failed with frequency {freq}: {e}")
            
            try:
                return TimeSeries.from_series(
                    data[column_name] if isinstance(data, pd.DataFrame) else data,
                    fill_missing_dates=False
                )
            except Exception as e:
                logger.warning(f"Failed without frequency: {e}")
                # Last resort: resample to daily
                series_data = data[column_name] if isinstance(data, pd.DataFrame) else data
                resampled = series_data.resample('D').ffill()
                return TimeSeries.from_series(resampled, freq='D')
        
        series_dict = {}
        
        if len(value_cols) == 1:
            # Univariate case with frequency handling
            series_dict['main'] = safe_create_ts(df_indexed, value_cols[0], detected_freq)
        else:
            # Multivariate case with frequency handling
            for col in value_cols:
                if col in df_indexed.columns:
                    try:
                        series_dict[col] = safe_create_ts(df_indexed, col, detected_freq)
                    except Exception as e:
                        logger.warning(f"Could not create series for {col}: {e}")
        
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
        
        # Convert to datetime and sort with robust parsing
        logger.info(f"Processing date column: {date_column}")
        logger.info(f"Sample date values: {df[date_column].head().tolist()}")
        
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce', format='mixed')
        
        # Remove rows with invalid dates
        before_count = len(df)
        df = df.dropna(subset=[date_column])
        after_count = len(df)
        
        if before_count != after_count:
            logger.warning(f"Removed {before_count - after_count} rows with invalid dates")
        
        if len(df) == 0:
            raise Exception(f"No valid dates found in column {date_column}")
        
        # Clean target column data
        logger.info(f"Processing target column: {target_column}")
        logger.info(f"Sample target values: {df[target_column].head().tolist()}")
        
        # Convert target to numeric and handle invalid values
        df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
        
        # Remove rows with invalid target values
        before_target_count = len(df)
        df = df.dropna(subset=[target_column])
        after_target_count = len(df)
        
        if before_target_count != after_target_count:
            logger.warning(f"Removed {before_target_count - after_target_count} rows with invalid target values")
        
        if len(df) == 0:
            raise Exception(f"No valid target values found in column {target_column}")
        
        df = df.sort_values(date_column).reset_index(drop=True)
        
        # Check for duplicate timestamps and aggregate them
        duplicate_count = df[date_column].duplicated().sum()
        if duplicate_count > 0:
            logger.warning(f"Found {duplicate_count} duplicate timestamps. Aggregating by mean...")
            # Group by date column and aggregate numeric columns by mean
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Create aggregation dict
            agg_dict = {col: 'mean' for col in numeric_cols if col in df.columns}
            
            # Group and aggregate - keep date_column for setting index later
            df = df.groupby(date_column, as_index=False).agg(agg_dict)
            logger.info(f"After aggregation: {len(df)} unique timestamps")
        
        # Sort again after aggregation
        df = df.sort_values(date_column).reset_index(drop=True)
        df_indexed = df.set_index(date_column)
        
        # Detect and set frequency - with robust handling for irregular data
        def detect_frequency(df_with_date_index):
            """Detect the most appropriate frequency for the time series"""
            try:
                # Try to infer frequency from the index
                inferred_freq = pd.infer_freq(df_with_date_index.index)
                if inferred_freq:
                    logger.info(f"Inferred frequency: {inferred_freq}")
                    return inferred_freq
                
                # If inference fails, analyze the data pattern
                time_diffs = df_with_date_index.index.to_series().diff().dropna()
                
                if len(time_diffs) == 0:
                    logger.warning("Not enough data points to detect frequency, using None")
                    return None
                
                most_common_diff = time_diffs.mode()
                
                if len(most_common_diff) > 0:
                    diff_days = most_common_diff.iloc[0].days
                    if diff_days == 1:
                        # Check if weekends are excluded (business days)
                        weekdays = df_with_date_index.index.weekday
                        has_weekends = any(day >= 5 for day in weekdays)  # 5=Saturday, 6=Sunday
                        if not has_weekends:
                            logger.info("Detected business day frequency (B)")
                            return 'B'  # Business day
                        else:
                            logger.info("Detected daily frequency (D)")
                            return 'D'  # Daily
                    elif diff_days == 7:
                        logger.info("Detected weekly frequency (W)")
                        return 'W'  # Weekly
                    elif diff_days >= 28 and diff_days <= 31:
                        logger.info("Detected monthly frequency (M)")
                        return 'M'  # Monthly
                
                # If we have irregular spacing, don't enforce a frequency
                logger.warning("Irregular spacing detected, using None for frequency")
                return None
                
            except Exception as e:
                logger.warning(f"Frequency detection failed: {e}, using None")
                return None
        
        detected_freq = detect_frequency(df_indexed)
        logger.info(f"Using frequency: {detected_freq}")
        
        # Helper function to safely create TimeSeries with frequency handling
        def safe_create_timeseries(data, column_name, freq):
            """Safely create TimeSeries, handling frequency issues"""
            try:
                # First try with detected frequency
                if freq is not None:
                    return TimeSeries.from_series(
                        data[column_name] if isinstance(data, pd.DataFrame) else data,
                        fill_missing_dates=True,
                        freq=freq
                    )
            except Exception as e:
                logger.warning(f"Failed with frequency {freq}: {e}, trying without frequency")
            
            try:
                # Try without frequency (let Darts infer)
                return TimeSeries.from_series(
                    data[column_name] if isinstance(data, pd.DataFrame) else data,
                    fill_missing_dates=False  # Don't fill missing dates if irregular
                )
            except Exception as e:
                logger.warning(f"Failed without frequency: {e}, trying with fill_missing_dates=True and freq=None")
            
            # Try with fill_missing_dates=True and freq=None (as suggested in error)
            try:
                return TimeSeries.from_series(
                    data[column_name] if isinstance(data, pd.DataFrame) else data,
                    fill_missing_dates=True,
                    freq=None
                )
            except Exception as e:
                logger.warning(f"Failed with fill_missing_dates=True: {e}, resampling to daily")
            
            # Last resort: resample to daily frequency
            try:
                series_data = data[column_name] if isinstance(data, pd.DataFrame) else data
                
                # Ensure we have a datetime index
                if not isinstance(series_data.index, pd.DatetimeIndex):
                    logger.warning(f"Index is not DatetimeIndex, cannot resample")
                    raise Exception(f"Cannot resample: index is not DatetimeIndex")
                
                # Check for duplicates in the resampled data
                if series_data.index.duplicated().any():
                    logger.warning(f"Found duplicates even after aggregation, using first occurrence")
                    series_data = series_data[~series_data.index.duplicated(keep='first')]
                
                # Resample to daily and forward fill
                resampled = series_data.resample('D').ffill()
                return TimeSeries.from_series(resampled, freq='D')
            except Exception as e:
                raise Exception(f"Could not create TimeSeries for {column_name}: {e}")
        
        # Helper function to safely create multivariate TimeSeries from DataFrame
        def safe_create_multivariate_timeseries(data, value_cols, freq):
            """Safely create multivariate TimeSeries for exogenous variables"""
            try:
                # First try with detected frequency
                if freq is not None:
                    return TimeSeries.from_dataframe(
                        data, 
                        value_cols=value_cols, 
                        fill_missing_dates=True, 
                        freq=freq
                    )
            except Exception as e:
                logger.warning(f"Multivariate creation failed with frequency {freq}: {e}, trying without frequency")
            
            try:
                # Try without frequency
                return TimeSeries.from_dataframe(
                    data, 
                    value_cols=value_cols, 
                    fill_missing_dates=False
                )
            except Exception as e:
                logger.warning(f"Multivariate creation failed without frequency: {e}, trying fill_missing_dates=True with freq=None")
            
            # Try with fill_missing_dates=True and freq=None
            try:
                return TimeSeries.from_dataframe(
                    data, 
                    value_cols=value_cols, 
                    fill_missing_dates=True, 
                    freq=None
                )
            except Exception as e:
                logger.warning(f"Multivariate creation failed: {e}, resampling to daily")
            
            # Last resort: resample each column and create TimeSeries
            try:
                resampled_data = data[value_cols].resample('D').ffill()
                return TimeSeries.from_dataframe(resampled_data, value_cols=value_cols, freq='D')
            except Exception as e:
                raise Exception(f"Could not create multivariate TimeSeries: {e}")
        
        # Prepare series based on forecasting type
        if forecasting_type == "univariate":
            target_series = safe_create_timeseries(df_indexed, target_column, detected_freq)
            train_target, val_target = target_series.split_after(train_split)
            
            # Validate we have enough data
            if len(train_target) < 10:
                raise Exception(f"Insufficient training data: only {len(train_target)} points. Need at least 10.")
            
            if len(val_target) < 1:
                raise Exception(f"Insufficient validation data: only {len(val_target)} points. Need at least 1. Try adjusting train_split or using more data.")
            
            logger.info(f"Univariate split: {len(train_target)} training points, {len(val_target)} validation points")
            
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
                try:
                    series_dict[col] = safe_create_timeseries(df_indexed, col, detected_freq)
                except Exception as e:
                    logger.warning(f"Could not create series for {col}: {e}")
            
            if not series_dict:
                raise Exception("Could not create any time series from the data")
            
            series_info = {
                "type": "multivariate",
                "series_dict": series_dict,
                "value_columns": list(series_dict.keys())
            }
            
        elif forecasting_type == "exogenous":
            target_series = safe_create_timeseries(df_indexed, target_column, detected_freq)
            
            exo_series = None
            if exogenous_columns and len(exogenous_columns) > 0:
                try:
                    exo_series = safe_create_multivariate_timeseries(df_indexed, exogenous_columns, detected_freq)
                except Exception as e:
                    logger.warning(f"Could not create exogenous series: {e}")
                    exo_series = None
            else:
                # Auto-select numeric columns as exogenous
                numeric_cols = [col for col in df.columns 
                              if col != date_column and col != target_column 
                              and col in df.select_dtypes(include=[np.number]).columns]
                if numeric_cols:
                    try:
                        exo_series = safe_create_multivariate_timeseries(df_indexed, numeric_cols, detected_freq)
                    except Exception as e:
                        logger.warning(f"Could not create auto-selected exogenous series: {e}")
                        exo_series = None
            
            train_target, val_target = target_series.split_after(train_split)
            
            # Validate we have enough data
            if len(train_target) < 10:
                raise Exception(f"Insufficient training data: only {len(train_target)} points. Need at least 10.")
            
            if len(val_target) < 1:
                raise Exception(f"Insufficient validation data: only {len(val_target)} points. Need at least 1. Try adjusting train_split or using more data.")
            
            logger.info(f"Data split: {len(train_target)} training points, {len(val_target)} validation points")
            
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
    
    logger.info(f"Training univariate models with {len(train_target)} training points, {len(val_target)} validation points")
    logger.info(f"Configuration: include_statistical={config.get('include_statistical', True)}, include_ml={config.get('include_ml', True)}, include_deep_learning={config.get('include_deep_learning', True)}")
    
    # Define model candidates
    candidates = {}
    
    # Statistical models
    if config.get("include_statistical", True):
        candidates.update({
            "NaiveSeasonal": NaiveSeasonal(K=config.get("seasonal_periods", 12)),
            "Theta": Theta(),
            "NaiveDrift": NaiveSeasonal(K=1),  # Simple drift model
        })
        
        # Add AutoARIMA (more robust than ARIMA)
        try:
            candidates["AutoARIMA"] = AutoARIMA(
                start_p=0, start_q=0, max_p=3, max_q=3,
                seasonal=True, stepwise=True, suppress_warnings=True,
                error_action="ignore", max_order=None, random_state=42
            )
        except Exception as e:
            logger.warning(f"AutoARIMA not available: {e}")
        
        # Add manual ARIMA models for better coverage
        try:
            from darts.models import ARIMA
            candidates.update({
                "ARIMA_101": ARIMA(p=1, d=0, q=1),
                "ARIMA_111": ARIMA(p=1, d=1, q=1),
                "ARIMA_212": ARIMA(p=2, d=1, q=2),
            })
        except Exception as e:
            logger.warning(f"Manual ARIMA models not available: {e}")
        
        # Add ExponentialSmoothing variations
        try:
            candidates.update({
                "ExponentialSmoothing": ExponentialSmoothing(),
                "ExponentialSmoothing_Add": ExponentialSmoothing(trend="add", seasonal="add"),
                "ExponentialSmoothing_Mul": ExponentialSmoothing(trend="mul", seasonal="mul"),
                "Holt": ExponentialSmoothing(trend="add", seasonal=None),
                "HoltWinters": ExponentialSmoothing(trend="add", seasonal="add", seasonal_periods=12),
            })
        except Exception as e:
            logger.warning(f"ExponentialSmoothing variations not available: {e}")
        
        # Add Statistical Ensemble Models
        try:
            from darts.models import StatsForecastAutoARIMA, StatsForecastAutoETS, StatsForecastAutoCES
            candidates.update({
                "StatsForecast_AutoARIMA": StatsForecastAutoARIMA(season_length=12),
                "StatsForecast_AutoETS": StatsForecastAutoETS(season_length=12),
                "StatsForecast_AutoCES": StatsForecastAutoCES(season_length=12),
            })
        except Exception as e:
            logger.warning(f"StatsForecast models not available: {e}")
        
        # Add Croston method for intermittent demand
        try:
            from darts.models import Croston
            candidates["Croston"] = Croston(version="classic")
        except Exception as e:
            logger.warning(f"Croston not available: {e}")
        
        # Add Prophet variations
        try:
            from darts.models import Prophet
            candidates.update({
                "Prophet": Prophet(),
                "Prophet_Weekly": Prophet(add_seasonalities={"name": "weekly", "period": 7, "fourier_order": 3}),
                "Prophet_Yearly": Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False),
            })
        except Exception as e:
            logger.warning(f"Prophet variations not available: {e}")
        
        # Add FFT (Fast Fourier Transform) model
        try:
            candidates["FFT"] = FFT(nr_freqs_to_keep=10, trend=None)
        except Exception as e:
            logger.warning(f"FFT not available: {e}")
        
        # Add Kalman Filter
        try:
            candidates["KalmanForecaster"] = KalmanForecaster(dim_x=4)
        except Exception as e:
            logger.warning(f"KalmanForecaster not available: {e}")
    
    # Machine Learning models
    if config.get("include_ml", True):
        try:
            # More robust configuration for ML models
            lags = min(24, len(train_target) // 3)  # Use up to 24 lags
            output_chunk = min(12, len(val_target))
            
            # Tree-based models (CPU efficient)
            candidates.update({
                "RandomForest": RandomForest(
                    lags=lags,
                    output_chunk_length=output_chunk,
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=1  # CPU friendly
                ),
                "RandomForest_Light": RandomForest(
                    lags=min(12, lags),
                    output_chunk_length=output_chunk,
                    n_estimators=50,  # Lighter version
                    max_depth=8,
                    random_state=42,
                    n_jobs=1
                ),
                "LightGBM": LightGBMModel(
                    lags=lags,
                    output_chunk_length=output_chunk,
                    random_state=42,
                    n_estimators=100,
                    max_depth=8,
                    num_leaves=31,
                    learning_rate=0.1
                ),
                "LightGBM_Fast": LightGBMModel(
                    lags=min(12, lags),
                    output_chunk_length=output_chunk,
                    random_state=42,
                    n_estimators=50,
                    max_depth=6,
                    num_leaves=15,
                    learning_rate=0.15
                ),
            })
            
            # Add XGBoost if available
            try:
                from darts.models import XGBModel
                candidates.update({
                    "XGBoost": XGBModel(
                        lags=lags,
                        output_chunk_length=output_chunk,
                        random_state=42,
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        n_jobs=1
                    ),
                    "XGBoost_Light": XGBModel(
                        lags=min(12, lags),
                        output_chunk_length=output_chunk,
                        random_state=42,
                        n_estimators=50,
                        max_depth=4,
                        learning_rate=0.15,
                        n_jobs=1
                    ),
                })
            except Exception as e:
                logger.warning(f"XGBoost not available: {e}")
            
            # Add CatBoost if available (CPU efficient)
            try:
                from darts.models import CatBoostModel
                candidates.update({
                    "CatBoost": CatBoostModel(
                        lags=lags,
                        output_chunk_length=output_chunk,
                        random_state=42,
                        iterations=100,
                        depth=6,
                        learning_rate=0.1,
                        verbose=False
                    ),
                })
            except Exception as e:
                logger.warning(f"CatBoost not available: {e}")
            
            # Linear models (very fast)
            try:
                from sklearn.linear_model import Ridge, Lasso, ElasticNet
                candidates.update({
                    "LinearRegression": LinearRegressionModel(
                        lags=lags,
                        output_chunk_length=output_chunk
                    ),
                    "Ridge": LinearRegressionModel(
                        lags=lags,
                        output_chunk_length=output_chunk,
                        model=Ridge(alpha=1.0)
                    ),
                    "Lasso": LinearRegressionModel(
                        lags=lags,
                        output_chunk_length=output_chunk,
                        model=Lasso(alpha=0.1)
                    ),
                    "ElasticNet": LinearRegressionModel(
                        lags=lags,
                        output_chunk_length=output_chunk,
                        model=ElasticNet(alpha=0.1, l1_ratio=0.5)
                    ),
                })
            except Exception as e:
                logger.warning(f"Linear models not available: {e}")
            
        except Exception as e:
            logger.warning(f"ML models not available: {e}")
    
    # Deep Learning models - CPU FRIENDLY
    if config.get("include_deep_learning", True):
        # Lower threshold for CPU-friendly models
        min_data_points = 50  # Reduced minimum for lighter models
        
        if len(train_target) >= min_data_points:
            try:
                # Calculate appropriate chunk lengths
                input_chunk = max(8, min(16, len(train_target) // 8))  # Smaller chunks for CPU
                output_chunk = max(4, min(8, len(val_target))) 
                
                # Moderate epochs for CPU training
                dl_epochs = max(min(max_epochs, 15), 10)  # 10-15 epochs max for CPU
                
                # Very lightweight models first
                candidates.update({
                    "NBEATS_Tiny": NBEATSModel(
                        input_chunk_length=input_chunk,
                        output_chunk_length=output_chunk,
                        n_epochs=dl_epochs,
                        num_stacks=2,  # Much smaller
                        num_blocks=1,
                        num_layers=2,
                        layer_widths=64,  # Smaller width
                        batch_size=32,
                        random_state=42,
                        pl_trainer_kwargs={
                            "enable_progress_bar": False,
                            "enable_model_summary": False,
                            "accelerator": "cpu",
                            "devices": 1
                        }
                    ),
                    "LSTM_Small": RNNModel(
                        model="LSTM",
                        hidden_dim=16,  # Small hidden size
                        n_rnn_layers=1,
                        input_chunk_length=input_chunk,
                        training_length=input_chunk + output_chunk,
                        n_epochs=dl_epochs,
                        dropout=0.2,
                        batch_size=32,
                        random_state=42,
                        pl_trainer_kwargs={
                            "enable_progress_bar": False,
                            "enable_model_summary": False,
                            "accelerator": "cpu",
                            "devices": 1
                        }
                    ),
                    "GRU_Small": RNNModel(
                        model="GRU",
                        hidden_dim=16,  # Small hidden size
                        n_rnn_layers=1,
                        input_chunk_length=input_chunk,
                        training_length=input_chunk + output_chunk,
                        n_epochs=dl_epochs,
                        dropout=0.2,
                        batch_size=32,
                        random_state=42,
                        pl_trainer_kwargs={
                            "enable_progress_bar": False,
                            "enable_model_summary": False,
                            "accelerator": "cpu",
                            "devices": 1
                        }
                    ),
                })
                
                # Add medium sized models if we have more data
                if len(train_target) >= 100:
                    candidates.update({
                        "NBEATS_Medium": NBEATSModel(
                            input_chunk_length=input_chunk,
                            output_chunk_length=output_chunk,
                            n_epochs=dl_epochs,
                            num_stacks=4,
                            num_blocks=1,
                            num_layers=2,
                            layer_widths=128,
                            batch_size=64,
                            random_state=42,
                            pl_trainer_kwargs={
                                "enable_progress_bar": False,
                                "enable_model_summary": False,
                                "accelerator": "cpu",
                                "devices": 1
                            }
                        ),
                        "LSTM_Medium": RNNModel(
                            model="LSTM",
                            hidden_dim=32,
                            n_rnn_layers=1,
                            input_chunk_length=input_chunk,
                            training_length=input_chunk + output_chunk,
                            n_epochs=dl_epochs,
                            dropout=0.1,
                            batch_size=64,
                            random_state=42,
                            pl_trainer_kwargs={
                                "enable_progress_bar": False,
                                "enable_model_summary": False,
                                "accelerator": "cpu",
                                "devices": 1
                            }
                        ),
                    })
                
                # Add Transformer-based models for larger datasets (still CPU friendly)
                if len(train_target) >= 200:
                    try:
                        from darts.models import TransformerModel
                        candidates.update({
                            "Transformer_Small": TransformerModel(
                                input_chunk_length=input_chunk,
                                output_chunk_length=output_chunk,
                                n_epochs=dl_epochs,
                                d_model=64,  # Small model dimension
                                nhead=4,  # Few attention heads
                                num_encoder_layers=2,
                                num_decoder_layers=2,
                                dim_feedforward=128,
                                dropout=0.1,
                                batch_size=32,
                                random_state=42,
                                pl_trainer_kwargs={
                                    "enable_progress_bar": False,
                                    "enable_model_summary": False,
                                    "accelerator": "cpu",
                                    "devices": 1
                                }
                            ),
                        })
                    except Exception as e:
                        logger.warning(f"Transformer model not available: {e}")
                
                # Add TCN (Temporal Convolutional Network) - very CPU efficient
                try:
                    from darts.models import TCNModel
                    candidates.update({
                        "TCN_Small": TCNModel(
                            input_chunk_length=input_chunk,
                            output_chunk_length=output_chunk,
                            n_epochs=dl_epochs,
                            num_filters=32,  # Small number of filters
                            num_layers=3,
                            dilation_base=2,
                            kernel_size=3,
                            dropout=0.2,
                            batch_size=32,
                            random_state=42,
                            pl_trainer_kwargs={
                                "enable_progress_bar": False,
                                "enable_model_summary": False,
                                "accelerator": "cpu",
                                "devices": 1
                            }
                        ),
                    })
                except Exception as e:
                    logger.warning(f"TCN model not available: {e}")
                
                logger.info(f"CPU-friendly deep learning enabled: input_chunk={input_chunk}, output_chunk={output_chunk}, epochs={dl_epochs}")
                
            except Exception as e:
                logger.warning(f"Deep learning models not available: {e}")
        else:
            logger.info(f"Skipping deep learning: need {min_data_points}+ points, have {len(train_target)}")
    
    # Log how many models we have before training
    logger.info(f"Total models to train: {len(candidates)}")
    if len(candidates) == 0:
        logger.error("No models available to train! Check configuration and dependencies.")
        return results
    
    # Train each model
    for name, model in candidates.items():
        try:
            logger.info(f"Training univariate model: {name}")
            
            import time
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                start_time = time.time()
                model.fit(train_target)
                pred = model.predict(len(val_target))
                training_time = time.time() - start_time
            
            # Validate predictions
            if pred is None or len(pred) == 0:
                raise Exception("Model returned empty predictions")
            
            # Calculate metrics with proper error handling
            try:
                smape_val = float(smape(val_target, pred))
                mae_val = float(mae(val_target, pred))
                rmse_val = float(rmse(val_target, pred))
                mape_val = float(mape(val_target, pred)) if len(val_target) > 0 else None
                
                # Validate metric values
                if not all(isinstance(x, (int, float)) and not (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) 
                          for x in [smape_val, mae_val, rmse_val] if x is not None):
                    raise Exception("Invalid metric values calculated")
                    
            except Exception as metric_error:
                raise Exception(f"Metric calculation failed: {metric_error}")
            
            # Calculate metrics
            metrics = {
                "model": name,
                "status": "ok",
                "smape": smape_val,
                "mae": mae_val,
                "rmse": rmse_val,
                "mape": mape_val,
                "training_time": round(training_time, 2),
                "error": None,
                "model_object": model
            }
            
            results.append(metrics)
            logger.info(f"✓ {name}: SMAPE={metrics['smape']:.4f}, MAE={metrics['mae']:.4f}, Time={metrics['training_time']:.2f}s")
            
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
                "training_time": 0.0,
                "error": error_msg
            })
    
    return results

async def train_multivariate_models(series_info: Dict, config: Dict, max_epochs: int) -> List[Dict]:
    """Train multivariate forecasting models"""
    results = []
    series_dict = series_info["series_dict"]
    value_columns = series_info["value_columns"]
    
    # First, try true multivariate models (VARIMA) on combined data
    if len(value_columns) > 1:
        try:
            # Create a single multivariate TimeSeries from all columns
            all_series = list(series_dict.values())
            if len(all_series) >= 2:
                # Combine series into multivariate TimeSeries
                combined_series = all_series[0]
                for i in range(1, len(all_series)):
                    combined_series = combined_series.stack(all_series[i])
                
                train_combined, val_combined = combined_series.split_after(config.get("train_split", 0.8))
                
                # Try VARIMA on combined multivariate series
                try:
                    logger.info("Training VARIMA on multivariate data")
                    varima_model = VARIMA(p=1, d=1, q=1)  # Simple configuration
                    
                    import time
                    start_time = time.time()
                    varima_model.fit(train_combined)
                    pred = varima_model.predict(len(val_combined))
                    training_time = time.time() - start_time
                    
                    # Calculate metrics on the first component (primary target)
                    val_first = val_combined.univariate_component(0)
                    pred_first = pred.univariate_component(0)
                    
                    metrics = {
                        "series": "multivariate_combined",
                        "model": "VARIMA",
                        "status": "ok",
                        "smape": float(smape(val_first, pred_first)),
                        "mae": float(mae(val_first, pred_first)),
                        "rmse": float(rmse(val_first, pred_first)),
                        "mape": float(mape(val_first, pred_first)) if len(val_first) > 0 else None,
                        "training_time": round(training_time, 2),
                        "error": None,
                        "model_object": varima_model
                    }
                    
                    results.append(metrics)
                    logger.info(f"✓ VARIMA: SMAPE={metrics['smape']:.4f}, MAE={metrics['mae']:.4f}")
                    
                except Exception as e:
                    error_msg = str(e)
                    logger.warning(f"✗ VARIMA failed: {error_msg}")
                    results.append({
                        "series": "multivariate_combined",
                        "model": "VARIMA",
                        "status": "failed",
                        "smape": None,
                        "mae": None,
                        "rmse": None,
                        "mape": None,
                        "training_time": 0.0,
                        "error": error_msg
                    })
        except Exception as e:
            logger.warning(f"Could not create combined multivariate series: {e}")
    
    # Train models for each individual series
    for series_name, series in series_dict.items():
        try:
            train, val = series.split_after(config.get("train_split", 0.8))
            
            candidates = {}
            
            # Add univariate statistical models
            candidates.update({
                "NaiveSeasonal": NaiveSeasonal(K=config.get("seasonal_periods", 12)),
                "NaiveDrift": NaiveSeasonal(K=1),
                "Theta": Theta(),
            })
            
            # Add more statistical models
            try:
                candidates.update({
                    "ExponentialSmoothing": ExponentialSmoothing(),
                    "Holt": ExponentialSmoothing(trend="add", seasonal=None),
                })
            except Exception as e:
                logger.warning(f"Additional statistical models not available: {e}")
            
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
                    
                    import time
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        
                        start_time = time.time()
                        model.fit(train)
                        pred = model.predict(len(val))
                        training_time = time.time() - start_time
                    
                    # Validate predictions
                    if pred is None or len(pred) == 0:
                        raise Exception("Model returned empty predictions")
                    
                    # Calculate metrics with error handling
                    try:
                        smape_val = float(smape(val, pred))
                        mae_val = float(mae(val, pred))
                        rmse_val = float(rmse(val, pred))
                        mape_val = float(mape(val, pred)) if len(val) > 0 else None
                        
                        # Validate metric values
                        if not all(isinstance(x, (int, float)) and not (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) 
                                  for x in [smape_val, mae_val, rmse_val] if x is not None):
                            raise Exception("Invalid metric values calculated")
                            
                    except Exception as metric_error:
                        raise Exception(f"Metric calculation failed: {metric_error}")
                    
                    metrics = {
                        "series": series_name,
                        "model": model_name,
                        "status": "ok",
                        "smape": smape_val,
                        "mae": mae_val,
                        "rmse": rmse_val,
                        "mape": mape_val,
                        "training_time": round(training_time, 2),
                        "error": None,
                        "model_object": model
                    }
                    
                    results.append(metrics)
                    logger.info(f"✓ {series_name}-{model_name}: SMAPE={metrics['smape']:.4f}, MAE={metrics['mae']:.4f}, Time={metrics['training_time']:.2f}s")
                    
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
                        "training_time": 0.0,
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
    
    logger.info(f"Training exogenous models with {len(train_target)} training points, {len(val_target)} validation points")
    logger.info(f"Exogenous variables available: {train_exo is not None}")
    
    # Validate data sizes
    if len(val_target) < 1:
        logger.error(f"Cannot train: validation set has {len(val_target)} points. Need at least 1.")
        return results
    
    # Define model candidates
    candidates = {}
    
    # Calculate appropriate lags and output chunk length
    lags = min(24, len(train_target) // 3)
    output_chunk = max(1, min(12, len(val_target)))  # Ensure at least 1
    lags_exo = [0, 1, 2, 3, 6, 12] if train_exo is not None else None
    
    logger.info(f"Model parameters: lags={lags}, output_chunk={output_chunk}, lags_exo={lags_exo}")
    
    # Statistical/ML models with exogenous support
    if config.get("include_statistical", True):
        try:
            candidates.update({
                "LinearRegression": LinearRegressionModel(
                    lags=lags,
                    lags_past_covariates=lags_exo,
                    output_chunk_length=output_chunk
                ),
            })
            
            # Add regularized linear models
            try:
                from sklearn.linear_model import Ridge, Lasso
                candidates.update({
                    "Ridge_Exo": LinearRegressionModel(
                        lags=lags,
                        lags_past_covariates=lags_exo,
                        output_chunk_length=output_chunk,
                        model=Ridge(alpha=1.0)
                    ),
                    "Lasso_Exo": LinearRegressionModel(
                        lags=lags,
                        lags_past_covariates=lags_exo,
                        output_chunk_length=output_chunk,
                        model=Lasso(alpha=0.1)
                    ),
                })
            except Exception as e:
                logger.warning(f"Regularized linear models not available: {e}")
                
        except Exception as e:
            logger.warning(f"LinearRegression not available: {e}")
    
    # Machine Learning models
    if config.get("include_ml", True):
        try:
            candidates.update({
                "RandomForest_Exo": RandomForest(
                    lags=lags,
                    lags_past_covariates=lags_exo if train_exo is not None else None,
                    output_chunk_length=output_chunk,
                    n_estimators=100,
                    random_state=42
                ),
                "LightGBM_Exo": LightGBMModel(
                    lags=lags,
                    lags_past_covariates=lags_exo if train_exo is not None else None,
                    output_chunk_length=output_chunk,
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
            
            import time
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                start_time = time.time()
                # FIX: All models should use past_covariates when available
                if train_exo is not None:
                    model.fit(train_target, past_covariates=train_exo)
                    pred = model.predict(len(val_target), past_covariates=val_exo)
                else:
                    # Fallback to univariate if no exogenous variables
                    model.fit(train_target)
                    pred = model.predict(len(val_target))
                training_time = time.time() - start_time
            
            # Validate predictions
            if pred is None or len(pred) == 0:
                raise Exception("Model returned empty predictions")
            
            # Calculate metrics with proper error handling
            try:
                smape_val = float(smape(val_target, pred))
                mae_val = float(mae(val_target, pred))
                rmse_val = float(rmse(val_target, pred))
                mape_val = float(mape(val_target, pred)) if len(val_target) > 0 else None
                
                # Validate metric values
                if not all(isinstance(x, (int, float)) and not (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) 
                          for x in [smape_val, mae_val, rmse_val] if x is not None):
                    raise Exception("Invalid metric values calculated")
                    
            except Exception as metric_error:
                raise Exception(f"Metric calculation failed: {metric_error}")
            
            metrics = {
                "model": name,
                "status": "ok",
                "smape": smape_val,
                "mae": mae_val,
                "rmse": rmse_val,
                "mape": mape_val,
                "training_time": round(training_time, 2),
                "exogenous_used": train_exo is not None,
                "error": None,
                "model_object": model
            }
            
            results.append(metrics)
            logger.info(f"✓ {name}: SMAPE={metrics['smape']:.4f}, MAE={metrics['mae']:.4f}, Time={metrics['training_time']:.2f}s")
            
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
                "training_time": 0.0,
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
        
        # Convert date column with robust parsing
        try:
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce', format='mixed')
            # Remove rows with invalid dates
            df = df.dropna(subset=[date_column])
            if len(df) == 0:
                raise HTTPException(status_code=400, detail=f"No valid dates found in {date_column}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Cannot convert {date_column} to datetime: {str(e)}")
        
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
        
        # Convert date column and sort with robust parsing
        try:
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce', format='mixed')
            # Remove rows with invalid dates
            df = df.dropna(subset=[date_column])
            if len(df) == 0:
                raise ValueError("No valid dates found")
            df = df.sort_values(date_column).reset_index(drop=True)
        except Exception as e:
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
            # Basic statistical models
            "NaiveSeasonal", "NaiveDrift", "Theta", "FFT", "KalmanForecaster",
            # ARIMA family
            "AutoARIMA", "ARIMA_101", "ARIMA_111", "ARIMA_212",
            # Exponential Smoothing family
            "ExponentialSmoothing", "ExponentialSmoothing_Add", "ExponentialSmoothing_Mul",
            "Holt", "HoltWinters",
            # Prophet family
            "Prophet", "Prophet_Weekly", "Prophet_Yearly",
            # StatsForecast models
            "StatsForecast_AutoARIMA", "StatsForecast_AutoETS", "StatsForecast_AutoCES",
            # Specialized models
            "Croston", "VARIMA"
        ],
        "machine_learning_models": [
            # Tree-based models
            "RandomForest", "RandomForest_Light", "LightGBM", "LightGBM_Fast",
            "XGBoost", "XGBoost_Light", "CatBoost",
            # Linear models
            "LinearRegression", "Ridge", "Lasso", "ElasticNet"
        ],
        "deep_learning_models": [
            # CPU-friendly deep learning
            "NBEATS_Tiny", "NBEATS_Medium", 
            "LSTM_Small", "LSTM_Medium",
            "GRU_Small",
            "TCN_Small", "Transformer_Small"
        ],
        "forecasting_types": [
            "univariate", "multivariate", "exogenous"
        ],
        "hardware_requirements": {
            "statistical_models": "Minimal CPU, <1GB RAM",
            "machine_learning_models": "Moderate CPU, 1-4GB RAM",
            "deep_learning_models": "CPU only, 2-8GB RAM, optimized for minimal hardware"
        },
        "performance_tiers": {
            "fastest": ["NaiveSeasonal", "NaiveDrift", "LinearRegression", "Ridge"],
            "fast": ["Theta", "ExponentialSmoothing", "Holt", "Lasso", "ElasticNet"],
            "moderate": ["AutoARIMA", "RandomForest_Light", "LightGBM_Fast", "Prophet"],
            "slower": ["NBEATS_Tiny", "LSTM_Small", "TCN_Small", "XGBoost"],
            "slowest": ["NBEATS_Medium", "LSTM_Medium", "Transformer_Small"]
        }
    }
    
    if not DARTS_AVAILABLE:
        models_info["error"] = "Darts library not available. Install with: pip install darts"
        models_info["fallback_models"] = [
            "Simple statistical models available through integrated ML training"
        ]
    
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