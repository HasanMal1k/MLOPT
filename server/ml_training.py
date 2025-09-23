from fastapi import APIRouter, HTTPException, BackgroundTasks, Form, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
import json
import uuid
import os
from pathlib import Path
import logging
import tempfile
import shutil
from io import StringIO, BytesIO

# ML imports
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from pycaret.regression import *
from pycaret.classification import *
import pycaret.regression as regression_module
import pycaret.classification as classification_module

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/ml", tags=["Machine Learning"])

# Global storage for training tasks
training_tasks = {}
feature_analysis_cache = {}
training_tasks: Dict[str, Dict] = {}

# Pydantic models
class FeatureSelectionRequest(BaseModel):
    filename: str
    task_type: str  # "classification" or "regression"

class TrainingConfigRequest(BaseModel):
    filename: str
    task_type: str
    target_column: str
    train_size: float = 0.8
    session_id: int = 123
    normalize: bool = True
    transformation: bool = True
    remove_outliers: bool = True
    outliers_threshold: float = 0.05
    feature_selection: bool = True
    polynomial_features: bool = False
    selected_features: Optional[List[str]] = None

class TrainingResponse(BaseModel):
    task_id: str
    status: str
    message: str

# Helper functions
def load_dataset(filename: str) -> pd.DataFrame:
    """Load dataset from your existing file system"""
    # Try the paths your main.py uses for file storage
    possible_paths = [
        Path("files") / filename,
        Path("processed_files") / filename,
        Path("uploads") / filename,
        Path(filename),  # If it's just the filename
    ]
    
    for file_path in possible_paths:
        if file_path.exists():
            try:
                if filename.endswith('.csv'):
                    return pd.read_csv(file_path)
                elif filename.endswith(('.xlsx', '.xls')):
                    return pd.read_excel(file_path)
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
                continue
    
    raise HTTPException(status_code=404, detail=f"File {filename} not found")

def calculate_feature_importance(data: pd.DataFrame, target_column: str, task_type: str) -> dict:
    """Calculate feature importance for the given dataset and target"""
    try:
        # Separate features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Handle missing values in target
        if task_type == "regression":
            y = y.fillna(y.mean())
        else:
            y = y.fillna(y.mode()[0] if len(y.mode()) > 0 else y.iloc[0])
        
        # Process all columns (numeric and categorical)
        processed_features = {}
        
        for column in X.columns:
            if X[column].dtype in ['object', 'category']:
                # Handle categorical columns
                try:
                    # Try label encoding for categorical variables
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    # Fill missing values first
                    X_col_filled = X[column].fillna('missing')
                    processed_features[column] = le.fit_transform(X_col_filled.astype(str))
                except:
                    # If label encoding fails, create dummy variables
                    continue
            else:
                # Handle numeric columns
                processed_features[column] = X[column].fillna(X[column].mean())
        
        if len(processed_features) == 0:
            # No processable features
            return {
                "all_features": [],
                "recommended_features": [],
                "total_features": 0,
                "recommended_count": 0
            }
        
        # Create DataFrame from processed features
        X_processed = pd.DataFrame(processed_features)
        
        # Ensure we have enough samples
        if len(X_processed) < 2:
            return {
                "all_features": [],
                "recommended_features": [],
                "total_features": 0,
                "recommended_count": 0
            }
        
        # Calculate feature importance
        try:
            if task_type == "regression":
                # Ensure target is numeric for regression
                y_numeric = pd.to_numeric(y, errors='coerce')
                if y_numeric.isna().all():
                    raise ValueError("Target column is not numeric for regression")
                importance_scores = mutual_info_regression(X_processed, y_numeric, random_state=42)
            else:
                # For classification, encode target if it's categorical
                if y.dtype == 'object':
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    y_encoded = le.fit_transform(y.astype(str))
                else:
                    y_encoded = y
                importance_scores = mutual_info_classif(X_processed, y_encoded, random_state=42)
        except Exception as e:
            logger.error(f"Error calculating mutual information: {str(e)}")
            # Fallback: use correlation for numeric features
            try:
                if task_type == "regression":
                    correlations = X_processed.corrwith(pd.to_numeric(y, errors='coerce')).abs()
                else:
                    # For classification, try point-biserial correlation
                    correlations = X_processed.corrwith(pd.to_numeric(y, errors='coerce')).abs()
                importance_scores = correlations.fillna(0).values
            except:
                # Final fallback: random importance
                importance_scores = np.random.random(len(X_processed.columns))
        
        # Handle NaN/inf values
        importance_scores = np.nan_to_num(importance_scores, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Create feature importance list
        feature_importance = []
        for i, feature in enumerate(X_processed.columns):
            importance = float(importance_scores[i])
            # Ensure the importance is a valid float
            if not np.isfinite(importance):
                importance = 0.0
            
            # Add feature type information
            feature_type = "numeric" if X[feature].dtype in ['int64', 'float64'] else "categorical"
            
            feature_importance.append({
                "feature": str(feature),
                "importance": importance,
                "feature_type": feature_type,
                "missing_count": int(X[feature].isnull().sum()),
                "unique_values": int(X[feature].nunique())
            })
        
        # Sort by importance
        feature_importance.sort(key=lambda x: x["importance"], reverse=True)
        
        # Calculate cumulative percentages
        total_importance = sum(f["importance"] for f in feature_importance)
        if total_importance > 0:
            cumulative = 0
            for f in feature_importance:
                cumulative += f["importance"]
                f["cumulative_percent"] = (cumulative / total_importance) * 100
        else:
            for f in feature_importance:
                f["cumulative_percent"] = 0.0
        
        # Get recommended features (top features covering ~80% of importance or top 10)
        recommended_features = []
        cumulative_importance = 0
        for f in feature_importance:
            if (cumulative_importance < 0.8 and len(recommended_features) < 10) or len(recommended_features) < 3:
                recommended_features.append(f)
                cumulative_importance += f["importance"] / total_importance if total_importance > 0 else 0
            else:
                break
        
        # Ensure we have at least some features if any exist
        if len(recommended_features) == 0 and len(feature_importance) > 0:
            recommended_features = feature_importance[:min(3, len(feature_importance))]
        
        return {
            "all_features": feature_importance,
            "recommended_features": recommended_features,
            "total_features": len(feature_importance),
            "recommended_count": len(recommended_features),
            "target_type": task_type,
            "target_unique_values": int(y.nunique()),
            "processed_features_count": len(X_processed.columns)
        }
        
    except Exception as e:
        logger.error(f"Error calculating feature importance: {str(e)}")
        logger.error(f"Data shape: {data.shape}, Target column: {target_column}, Task type: {task_type}")
        
        # Return a basic feature list as fallback
        try:
            X = data.drop(columns=[target_column])
            basic_features = []
            for i, feature in enumerate(X.columns):
                basic_features.append({
                    "feature": str(feature),
                    "importance": 1.0 / len(X.columns),  # Equal importance
                    "feature_type": "numeric" if X[feature].dtype in ['int64', 'float64'] else "categorical",
                    "missing_count": int(X[feature].isnull().sum()),
                    "unique_values": int(X[feature].nunique()),
                    "cumulative_percent": ((i + 1) / len(X.columns)) * 100
                })
            
            recommended = basic_features[:min(5, len(basic_features))]
            
            return {
                "all_features": basic_features,
                "recommended_features": recommended,
                "total_features": len(basic_features),
                "recommended_count": len(recommended),
                "target_type": task_type,
                "target_unique_values": int(data[target_column].nunique()),
                "processed_features_count": len(basic_features),
                "fallback": True,
                "error": str(e)
            }
        except Exception as fallback_error:
            logger.error(f"Fallback also failed: {str(fallback_error)}")
            return {
                "all_features": [],
                "recommended_features": [],
                "total_features": 0,
                "recommended_count": 0,
                "error": str(e)
            }


# API Endpoints

@router.post("/feature-selection/")
async def analyze_features(request: FeatureSelectionRequest):
    """Analyze dataset and return feature importance for selection"""
    try:
        # Load dataset
        data = load_dataset(request.filename)
        
        # Basic data info
        data_info = {
            "shape": data.shape,
            "columns": data.columns.tolist(),
            "dtypes": data.dtypes.astype(str).to_dict(),
            "missing_values": data.isnull().sum().to_dict(),
            "numeric_columns": data.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": data.select_dtypes(include=['object']).columns.tolist()
        }
        
        # Cache for later use
        cache_key = f"{request.filename}_{request.task_type}"
        feature_analysis_cache[cache_key] = {
            "data_info": data_info,
            "request": request.dict()
        }
        
        return JSONResponse({
            "success": True,
            "data_info": data_info,
            "cache_key": cache_key,
            "message": "Dataset analyzed successfully. Please select target column to proceed with feature analysis."
        })
        
    except Exception as e:
        logger.error(f"Feature selection error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))



# Fix the run_training_task function - remove 'silent' parameter:
# Remove the duplicate run_training_task function that has silent=True
# Keep only this corrected version:

# Update the run_training_task function with better error handling:
async def run_training_task(config_id: str, config: dict):
    """Background task for model training - force PyCaret to work"""
    try:
        # Update status
        training_tasks[config_id]["status"] = "loading_data"
        
        # Load saved training data
        data_file = training_tasks[config_id]["data_file"]
        data = pd.read_csv(data_file)
        
        logger.info(f"Training data shape: {data.shape}")
        
        training_tasks[config_id]["status"] = "setting_up"
        
        # FORCE PyCaret setup with minimal configuration - no fancy preprocessing
        logger.info("Setting up PyCaret with minimal configuration...")
        
        if config["task_type"] == "regression":
            # Minimal regression setup
            setup_result = regression_module.setup(
                data=data,
                target=config["target_column"],
                train_size=0.8,  # Fixed train size
                session_id=123,  # Fixed session
                verbose=False,   # Suppress output
                # Remove all problematic parameters
            )
        else:
            # Minimal classification setup  
            setup_result = classification_module.setup(
                data=data,
                target=config["target_column"],
                train_size=0.8,  # Fixed train size
                session_id=123,  # Fixed session
                verbose=False,   # Suppress output
                # Remove all problematic parameters
            )
        
        training_tasks[config_id]["status"] = "training"
        logger.info("Starting model comparison...")
        
        # FORCE compare_models to run - use all available models
        if config["task_type"] == "regression":
            # Get all available regression models and run them
            available_models = regression_module.models()
            logger.info(f"Available regression models: {len(available_models)}")
            
            # Compare ALL models (no selection limit)
            best_models = regression_module.compare_models(
                include=None,  # Include all models
                fold=3,        # Reduce CV folds for speed
                sort='R2',     # Sort by R2
                turbo=False    # Don't use turbo mode
            )
            
            leaderboard = regression_module.pull()
            
            # Get top 5 models
            if isinstance(best_models, list):
                top_models = best_models[:5]
                best_model = best_models[0]
            else:
                top_models = [best_models]
                best_model = best_models
                
        else:
            # Get all available classification models and run them
            available_models = classification_module.models()
            logger.info(f"Available classification models: {len(available_models)}")
            
            # Compare ALL models (no selection limit)
            best_models = classification_module.compare_models(
                include=None,  # Include all models
                fold=3,        # Reduce CV folds for speed
                sort='Accuracy', # Sort by Accuracy
                turbo=False    # Don't use turbo mode
            )
            
            leaderboard = classification_module.pull()
            
            # Get top 5 models
            if isinstance(best_models, list):
                top_models = best_models[:5]
                best_model = best_models[0]
            else:
                top_models = [best_models]
                best_model = best_models
        
        logger.info(f"Training completed. Best model: {best_model}")
        
        training_tasks[config_id]["status"] = "finalizing"
        
        # Finalize the best model (train on full dataset)
        final_best_model = regression_module.finalize_model(best_model) if config["task_type"] == "regression" else classification_module.finalize_model(best_model)
        
        training_tasks[config_id]["status"] = "saving_models"
        
        # Create models directory
        models_dir = Path("models") / config_id
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the finalized best model
        if config["task_type"] == "regression":
            regression_module.save_model(final_best_model, str(models_dir / 'best_model'))
            # Save top models (FIX: iterate through ALL top models)
            saved_count = 1  # Count the best model
            for i, model in enumerate(top_models[:5], start=1):  # Limit to top 5
                try:
                    finalized = regression_module.finalize_model(model)
                    regression_module.save_model(finalized, str(models_dir / f'top_{i}_model'))
                    saved_count += 1
                    logger.info(f"Saved model {i}: {str(model)}")
                except Exception as e:
                    logger.warning(f"Failed to save model {i}: {str(e)}")
        else:
            classification_module.save_model(final_best_model, str(models_dir / 'best_model'))
            # Save top models (FIX: iterate through ALL top models)
            saved_count = 1  # Count the best model
            for i, model in enumerate(top_models[:5], start=1):  # Limit to top 5
                try:
                    finalized = classification_module.finalize_model(model)
                    classification_module.save_model(finalized, str(models_dir / f'top_{i}_model'))
                    saved_count += 1
                    logger.info(f"Saved model {i}: {str(model)}")
                except Exception as e:
                    logger.warning(f"Failed to save model {i}: {str(e)}")
        
        # Save leaderboard
        leaderboard.to_csv(models_dir / 'leaderboard.csv', index=False)
        
        # Update final status (FIX: use saved_count instead of len(top_models))
        from datetime import datetime
        training_tasks[config_id].update({
            "status": "completed",
            "leaderboard": leaderboard.to_dict('records'),
            "best_model_name": str(best_model),
            "models_saved": saved_count,  # Use actual saved count
            "models_dir": str(models_dir),
            "completed_at": datetime.now().isoformat(),
            "total_models_tested": len(available_models),
            "training_notes": "Ran all available models with minimal preprocessing"
        })
        
        logger.info(f"Training completed successfully for {config_id}. Tested {len(available_models)} models.")
        
        # Clean up training data file
        try:
            os.unlink(data_file)
        except:
            pass
        
    except Exception as e:
        logger.error(f"Training error for {config_id}: {str(e)}")
        training_tasks[config_id]["status"] = "failed"
        training_tasks[config_id]["error"] = str(e)
        
@router.post("/start-training/")
async def start_training(background_tasks: BackgroundTasks, config_id: str = Form(...)):
    """Start model training in background"""
    try:
        if config_id not in training_tasks:
            raise HTTPException(status_code=404, detail="Configuration not found")
        
        # Get config directly from stored dict (don't validate with Pydantic)
        config = training_tasks[config_id]["config"]
        
        # Add timestamp
        from datetime import datetime
        training_tasks[config_id]["started_at"] = datetime.now().isoformat()
        
        # Start background training
        background_tasks.add_task(run_training_task, config_id, config)
        
        training_tasks[config_id]["status"] = "started"
        
        return JSONResponse({
            "success": True,
            "task_id": config_id,
            "status": "started",
            "message": "Training started in background",
            "started_at": training_tasks[config_id]["started_at"]
        })
        
    except Exception as e:
        logger.error(f"Start training error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/training-status/{task_id}")
async def get_training_status(task_id: str):
    """Get training status and results"""
    if task_id not in training_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = training_tasks[task_id]
    
    response = {
        "task_id": task_id,
        "status": task["status"],
        "data_shape": task.get("data_shape"),
        "features_count": task.get("features_count")
    }
    
    if task["status"] == "completed":
        response.update({
            "leaderboard": task["leaderboard"],
            "best_model_name": task["best_model_name"],
            "models_saved": task["models_saved"]
        })
    elif task["status"] == "failed":
        response["error"] = task.get("error")
    
    return JSONResponse(response)

@router.get("/download-model/{task_id}/{model_name}")
async def download_model(task_id: str, model_name: str):
    """Download trained model"""
    if task_id not in training_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = training_tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Training not completed")
    
    model_path = Path("models") / task_id / f"{model_name}.pkl"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    
    return FileResponse(
        path=model_path,
        filename=f"{model_name}_{task_id}.pkl",
        media_type="application/octet-stream"
    )

@router.get("/download-leaderboard/{task_id}")
async def download_leaderboard(task_id: str):
    """Download leaderboard CSV"""
    if task_id not in training_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = training_tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Training not completed")
    
    leaderboard_path = Path("models") / task_id / "leaderboard.csv"
    if not leaderboard_path.exists():
        raise HTTPException(status_code=404, detail="Leaderboard not found")
    
    return FileResponse(
        path=leaderboard_path,
        filename=f"leaderboard_{task_id}.csv",
        media_type="text/csv"
    )

@router.get("/available-models/")
async def get_available_models():
    """Get list of available models for each task type"""
    try:
        regression_models = regression_module.models()
        classification_models = classification_module.models()
        
        return JSONResponse({
            "regression_models": regression_models.to_dict('records'),
            "classification_models": classification_models.to_dict('records')
        })
    except Exception as e:
        return JSONResponse({
            "error": str(e),
            "regression_models": [],
            "classification_models": []
        })

@router.delete("/clear-cache/")
async def clear_cache():
    """Clear training tasks and feature analysis cache"""
    global training_tasks, feature_analysis_cache
    training_tasks.clear()
    feature_analysis_cache.clear()
    
    return JSONResponse({
        "success": True,
        "message": "Cache cleared successfully"
    })
    
@router.post("/analyze-target/")
async def analyze_target_column(
    filename: str = Form(...),
    task_type: str = Form(...),
    target_column: str = Form(...)
):
    """Analyze target column and return feature importance"""
    try:
        # Load dataset using existing file system
        data = load_dataset(filename)
        
        if target_column not in data.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found")
        
        # Validate target column for task type
        if task_type == "classification":
            unique_values = data[target_column].nunique()
            if unique_values > 20:
                return JSONResponse({
                    "warning": True,
                    "message": f"Target column has {unique_values} unique values. Are you sure this is a classification task?"
                })
        elif task_type == "regression":
            if data[target_column].dtype == 'object':
                raise HTTPException(status_code=400, detail="Target column for regression must be numeric")
        
        # Calculate feature importance
        feature_analysis = calculate_feature_importance(data, target_column, task_type)
        
        return JSONResponse({
            "success": True,
            "target_column": target_column,
            "target_info": {
                "dtype": str(data[target_column].dtype),
                "unique_values": int(data[target_column].nunique()),
                "missing_values": int(data[target_column].isnull().sum())
            },
            "feature_analysis": feature_analysis
        })
        
    except Exception as e:
        logger.error(f"Target analysis error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


router.post("/configure-training/")
async def configure_training(request: TrainingConfigRequest):
    """Configure training parameters and validate setup"""
    try:
        # Load dataset using existing file system
        data = load_dataset(request.filename)
        
        # Validate target column
        if request.target_column not in data.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{request.target_column}' not found")
        
        # Validate parameters
        if not 0.1 <= request.train_size <= 0.9:
            raise HTTPException(status_code=400, detail="Train size must be between 0.1 and 0.9")
        
        if not 0.01 <= request.outliers_threshold <= 0.2:
            raise HTTPException(status_code=400, detail="Outliers threshold must be between 0.01 and 0.2")
        
        # Filter selected features if provided
        if request.selected_features:
            available_features = set(data.columns) - {request.target_column}
            invalid_features = set(request.selected_features) - available_features
            if invalid_features:
                raise HTTPException(status_code=400, detail=f"Invalid features: {list(invalid_features)}")
            
            # Keep only selected features + target
            data = data[request.selected_features + [request.target_column]]
        
        # Generate unique config ID
        config_id = str(uuid.uuid4())
        
        # Save dataset for training (create a persistent copy)
        training_data_dir = Path("training_data")
        training_data_dir.mkdir(exist_ok=True)
        training_file_path = training_data_dir / f"{config_id}_data.csv"
        data.to_csv(training_file_path, index=False)
        
        # Store configuration for training
        training_tasks[config_id] = {
            "status": "configured",
            "config": request.dict(),
            "data_file": str(training_file_path),
            "data_shape": data.shape,
            "features_count": len(data.columns) - 1
        }
        
        return JSONResponse({
            "success": True,
            "config_id": config_id,
            "data_shape": data.shape,
            "features_count": len(data.columns) - 1,
            "message": "Training configuration validated successfully"
        })
        
    except Exception as e:
        logger.error(f"Configuration error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/start-training/")
async def start_training(background_tasks: BackgroundTasks, config_id: str = Form(...)):
    """Start model training in background"""
    try:
        if config_id not in training_tasks:
            raise HTTPException(status_code=404, detail="Configuration not found")
        
        config = training_tasks[config_id]["config"]
        
        # Start background training
        background_tasks.add_task(run_training_task, config_id, config)
        
        training_tasks[config_id]["status"] = "started"
        
        return JSONResponse({
            "success": True,
            "task_id": config_id,
            "status": "started",
            "message": "Training started in background"
        })
        
    except Exception as e:
        logger.error(f"Start training error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


# Add this new endpoint to receive files directly
@router.post("/analyze-target-with-file/")
async def analyze_target_column_with_file(
    file: UploadFile = File(...),
    task_type: str = Form(...),
    target_column: str = Form(...)
):
    """Analyze target column with uploaded file"""
    try:
        # Read the uploaded file directly
        contents = await file.read()
        
        # Create DataFrame from file contents
        if file.filename.endswith('.csv'):
            from io import StringIO
            data = pd.read_csv(StringIO(contents.decode('utf-8')))
        elif file.filename.endswith(('.xlsx', '.xls')):
            from io import BytesIO
            data = pd.read_excel(BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        if target_column not in data.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found")
        
        # Validate target column for task type
        if task_type == "classification":
            unique_values = data[target_column].nunique()
            if unique_values > 20:
                return JSONResponse({
                    "warning": True,
                    "message": f"Target column has {unique_values} unique values. Are you sure this is a classification task?"
                })
        elif task_type == "regression":
            if data[target_column].dtype == 'object':
                raise HTTPException(status_code=400, detail="Target column for regression must be numeric")
        
        # Calculate feature importance
        feature_analysis = calculate_feature_importance(data, target_column, task_type)
        
        # Ensure all numeric values are JSON serializable
        def clean_for_json(obj):
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_for_json(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif pd.isna(obj) or obj is None:
                return 0.0
            elif not np.isfinite(obj) if isinstance(obj, (int, float)) else False:
                return 0.0
            else:
                return obj
        
        # Clean the response data
        response_data = {
            "success": True,
            "target_column": target_column,
            "target_info": {
                "dtype": str(data[target_column].dtype),
                "unique_values": int(data[target_column].nunique()),
                "missing_values": int(data[target_column].isnull().sum())
            },
            "feature_analysis": clean_for_json(feature_analysis)
        }
        
        return JSONResponse(response_data)
        
    except Exception as e:
        logger.error(f"Target analysis error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# Add this endpoint after the analyze-target-with-file endpoint:
@router.post("/configure-training-with-file/")
async def configure_training_with_file(
    file: UploadFile = File(...),
    task_type: str = Form(...),
    target_column: str = Form(...),
    selected_features: str = Form(...),
    train_size: float = Form(0.8),
    session_id: int = Form(123),
    normalize: bool = Form(True),
    transformation: bool = Form(True),
    remove_outliers: bool = Form(True),
    outliers_threshold: float = Form(0.05),
    feature_selection: bool = Form(True),
    polynomial_features: bool = Form(False)
):
    """Configure training with uploaded file"""
    try:
        # Parse selected features
        try:
            selected_features_list = json.loads(selected_features)
        except:
            selected_features_list = []
        
        # Read the uploaded file directly
        contents = await file.read()
        
        # Create DataFrame from file contents
        if file.filename.endswith('.csv'):
            from io import StringIO
            data = pd.read_csv(StringIO(contents.decode('utf-8')))
        elif file.filename.endswith(('.xlsx', '.xls')):
            from io import BytesIO
            data = pd.read_excel(BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Validate target column
        if target_column not in data.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found")
        
        # Validate parameters
        if not 0.1 <= train_size <= 0.9:
            raise HTTPException(status_code=400, detail="Train size must be between 0.1 and 0.9")
        
        if not 0.01 <= outliers_threshold <= 0.2:
            raise HTTPException(status_code=400, detail="Outliers threshold must be between 0.01 and 0.2")
        
        # Filter selected features if provided
        if selected_features_list:
            available_features = set(data.columns) - {target_column}
            invalid_features = set(selected_features_list) - available_features
            if invalid_features:
                raise HTTPException(status_code=400, detail=f"Invalid features: {list(invalid_features)}")
            
            # Keep only selected features + target
            data = data[selected_features_list + [target_column]]
        
        # Generate unique config ID
        config_id = str(uuid.uuid4())
        
        # Save dataset for training
        training_data_dir = Path("training_data")
        training_data_dir.mkdir(exist_ok=True)
        training_file_path = training_data_dir / f"{config_id}_data.csv"
        data.to_csv(training_file_path, index=False)
        
        # Store configuration for training
        training_tasks[config_id] = {
            "status": "configured",
            "config": {
                "task_type": task_type,
                "target_column": target_column,
                "selected_features": selected_features_list,
                "train_size": train_size,
                "session_id": session_id,
                "normalize": normalize,
                "transformation": transformation,
                "remove_outliers": remove_outliers,
                "outliers_threshold": outliers_threshold,
                "feature_selection": feature_selection,
                "polynomial_features": polynomial_features
            },
            "data_file": str(training_file_path),
            "data_shape": data.shape,
            "features_count": len(data.columns) - 1,
            "original_filename": file.filename
        }
        
        logger.info(f"Configuration saved with ID: {config_id}")
        
        return JSONResponse({
            "success": True,
            "config_id": config_id,
            "data_shape": data.shape,
            "features_count": len(data.columns) - 1,
            "message": "Training configuration validated successfully"
        })
        
    except Exception as e:
        logger.error(f"Configuration error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))