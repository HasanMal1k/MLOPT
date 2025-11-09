"""
Model Management API - Save, List, Download trained models
"""
from fastapi import APIRouter, HTTPException, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import pickle
import json
from pathlib import Path
from datetime import datetime
import logging
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/models", tags=["Model Management"])

# Try to import Supabase (optional dependency)
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    logger.warning("⚠️ Supabase module not installed. Model saving features will be disabled.")
    logger.warning("   To enable: pip install supabase")

# Supabase client (only if available)
supabase = None
if SUPABASE_AVAILABLE:
    SUPABASE_URL = os.getenv("SUPABASE_URL", "")
    SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")  # Service role key (bypasses RLS)
    SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")  # Fallback to anon key
    
    # Prefer service key for backend operations (bypasses RLS)
    key_to_use = SUPABASE_SERVICE_KEY if SUPABASE_SERVICE_KEY else SUPABASE_KEY
    
    if SUPABASE_URL and key_to_use:
        try:
            supabase = create_client(SUPABASE_URL, key_to_use)
            if SUPABASE_SERVICE_KEY:
                logger.info("✅ Supabase client initialized with SERVICE ROLE key (RLS bypassed)")
            else:
                logger.warning("⚠️ Using ANON key - RLS policies will apply. For backend operations, set SUPABASE_SERVICE_KEY")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Supabase client: {e}")
            supabase = None
    else:
        logger.warning("⚠️ Supabase credentials not configured. Set SUPABASE_URL and SUPABASE_SERVICE_KEY env variables.")


# Models directory
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Pydantic models
class SaveModelRequest(BaseModel):
    task_id: str
    model_name: str
    description: Optional[str] = None
    tags: Optional[List[str]] = []

class ModelMetadata(BaseModel):
    id: str
    model_name: str
    model_type: str
    algorithm: str
    metrics: Dict[str, Any]
    file_size: int
    created_at: str
    description: Optional[str] = None
    tags: Optional[List[str]] = []

class ModelListResponse(BaseModel):
    models: List[ModelMetadata]
    total_count: int


@router.post("/save")
async def save_trained_model(request: SaveModelRequest, user_id: str):
    """
    Save a trained model to Supabase storage and database
    
    Args:
        request: SaveModelRequest with task_id, model_name, description, tags
        user_id: User ID from authentication (passed from frontend)
    
    Returns:
        Success message with model_id
    """
    
    # Check if Supabase is available
    if not SUPABASE_AVAILABLE or not supabase:
        raise HTTPException(
            status_code=503, 
            detail="Model saving is not configured. Supabase module not installed or credentials not set."
        )
    
    try:
        logger.info(f"📦 Saving model for task {request.task_id}, user {user_id}")
        
        # Import training tasks from ml_training
        from ml_training import training_tasks
        
        # Check if task exists
        if request.task_id not in training_tasks:
            raise HTTPException(status_code=404, detail="Training task not found")
        
        task = training_tasks[request.task_id]
        
        # Check if training is completed
        if task["status"] != "completed":
            raise HTTPException(status_code=400, detail=f"Training not completed. Status: {task['status']}")
        
        # Get the best model information
        leaderboard = task.get("leaderboard", [])
        if not leaderboard:
            raise HTTPException(status_code=404, detail="No models found in training results")
        
        best_model = leaderboard[0]  # Best model is first in sorted leaderboard
        model_type = task.get("task_type", "unknown")
        
        # Find the model file in models directory
        models_dir = task.get("models_dir")
        if not models_dir:
            # Try to find models directory
            models_dir = MODELS_DIR / request.task_id
        else:
            models_dir = Path(models_dir)
        
        # Look for the best model file
        model_file = models_dir / "best_model.pkl"
        if not model_file.exists():
            # Try alternative naming
            model_file = models_dir / f"{best_model['Model'].replace(' ', '_').lower()}.pkl"
        
        if not model_file.exists():
            raise HTTPException(status_code=404, detail=f"Model file not found at {model_file}")
        
        # Read model file
        with open(model_file, 'rb') as f:
            model_data = f.read()
        
        file_size = len(model_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create storage path: user_id/model_name_timestamp.pkl
        storage_filename = f"{user_id}/{request.model_name.replace(' ', '_')}_{timestamp}.pkl"
        
        # Upload to Supabase Storage
        if supabase:
            try:
                logger.info(f"☁️ Uploading to Supabase Storage: {storage_filename}")
                
                # Upload to model-files bucket
                storage_response = supabase.storage.from_('model-files').upload(
                    storage_filename,
                    model_data,
                    file_options={"content-type": "application/octet-stream"}
                )
                
                logger.info(f"✅ Model file uploaded to storage")
                
                # Prepare model metadata
                feature_columns = task.get("config", {}).get("selected_features", [])
                target_column = task.get("config", {}).get("target_column")
                training_config = task.get("config", {})
                
                # Extract metrics from best model
                metrics = {
                    k: v for k, v in best_model.items() 
                    if k not in ['Model', 'TT (Sec)', 'status']
                }
                
                # Prepare database record
                model_record = {
                    "user_id": user_id,
                    "file_id": task.get("file_id"),  # Reference to dataset
                    "model_name": request.model_name,
                    "model_type": model_type,
                    "algorithm": best_model.get("Model", "Unknown"),
                    "metrics": metrics,
                    "training_config": {
                        "train_size": training_config.get("train_size", 0.8),
                        "normalize": training_config.get("normalize", True),
                        "transformation": training_config.get("transformation", True),
                        "feature_selection": training_config.get("feature_selection", True),
                        "cv_folds": training_config.get("cv_folds", 3),
                    },
                    "model_file_path": storage_filename,
                    "model_file_size": file_size,
                    "feature_columns": feature_columns,
                    "target_column": target_column,
                    "preprocessing_steps": training_config.get("preprocessing_steps", {}),
                    "training_time_seconds": best_model.get("TT (Sec)", 0),
                    "training_samples": task.get("training_samples"),
                    "test_samples": task.get("test_samples"),
                    "status": "ready",
                    "description": request.description,
                    "tags": request.tags or []
                }
                
                # Insert into database
                logger.info(f"💾 Saving model metadata to database")
                db_response = supabase.table('trained_models').insert(model_record).execute()
                
                model_id = db_response.data[0]['id']
                
                logger.info(f"✅ Model saved successfully! ID: {model_id}")
                
                return JSONResponse({
                    "success": True,
                    "message": "Model saved successfully",
                    "model_id": model_id,
                    "model_name": request.model_name,
                    "file_size": file_size,
                    "storage_path": storage_filename
                })
                
            except Exception as storage_error:
                logger.error(f"❌ Supabase error: {storage_error}")
                raise HTTPException(status_code=500, detail=f"Failed to save model: {str(storage_error)}")
        else:
            raise HTTPException(status_code=500, detail="Supabase not configured")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error saving model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_user_models(user_id: str, status: Optional[str] = None):
    """
    List all saved models for a user
    
    Args:
        user_id: User ID from authentication
        status: Optional filter by status (ready, training, failed)
    
    Returns:
        List of user's models with metadata
    """
    
    if not SUPABASE_AVAILABLE or not supabase:
        raise HTTPException(
            status_code=503,
            detail="Model listing is not configured. Supabase module not installed or credentials not set."
        )
    
    try:
        if not supabase:
            raise HTTPException(status_code=500, detail="Supabase not configured")
        
        logger.info(f"📋 Fetching models for user {user_id}")
        
        # Build query
        query = supabase.table('trained_models').select('*').eq('user_id', user_id)
        
        if status:
            query = query.eq('status', status)
        
        # Execute query with ordering
        response = query.order('created_at', desc=True).execute()
        
        models = response.data
        
        logger.info(f"✅ Found {len(models)} models")
        
        return JSONResponse({
            "success": True,
            "models": models,
            "total_count": len(models)
        })
        
    except Exception as e:
        logger.error(f"❌ Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download/{model_id}")
async def download_model(model_id: str, user_id: str):
    """
    Generate a signed URL to download a model file
    
    Args:
        model_id: Model ID to download
        user_id: User ID for security check
    
    Returns:
        Signed URL for downloading the model file
    """
    
    if not SUPABASE_AVAILABLE or not supabase:
        raise HTTPException(
            status_code=503,
            detail="Model download is not configured. Supabase module not installed or credentials not set."
        )
    
    try:
        if not supabase:
            raise HTTPException(status_code=500, detail="Supabase not configured")
        
        logger.info(f"⬇️ Generating download URL for model {model_id}")
        
        # Get model metadata
        response = supabase.table('trained_models').select('*').eq('id', model_id).eq('user_id', user_id).single().execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Model not found or access denied")
        
        model = response.data
        model_file_path = model['model_file_path']
        
        # Generate signed URL (valid for 1 hour)
        signed_url_response = supabase.storage.from_('model-files').create_signed_url(
            model_file_path,
            expires_in=3600  # 1 hour
        )
        
        if 'signedURL' not in signed_url_response:
            raise HTTPException(status_code=500, detail="Failed to generate download URL")
        
        # Update last downloaded timestamp
        supabase.table('trained_models').update({
            'last_downloaded_at': datetime.now().isoformat()
        }).eq('id', model_id).execute()
        
        logger.info(f"✅ Download URL generated")
        
        return JSONResponse({
            "success": True,
            "download_url": signed_url_response['signedURL'],
            "model_name": model['model_name'],
            "file_size": model['model_file_size'],
            "expires_in": 3600
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error generating download URL: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete/{model_id}")
async def delete_model(model_id: str, user_id: str):
    """
    Delete a saved model (soft delete - marks as deleted)
    
    Args:
        model_id: Model ID to delete
        user_id: User ID for security check
    
    Returns:
        Success message
    """
    
    if not SUPABASE_AVAILABLE or not supabase:
        raise HTTPException(
            status_code=503,
            detail="Model deletion is not configured. Supabase module not installed or credentials not set."
        )
    
    try:
        if not supabase:
            raise HTTPException(status_code=500, detail="Supabase not configured")
        
        logger.info(f"🗑️ Deleting model {model_id}")
        
        # Verify ownership
        response = supabase.table('trained_models').select('*').eq('id', model_id).eq('user_id', user_id).single().execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Model not found or access denied")
        
        # Soft delete - mark as deleted
        supabase.table('trained_models').update({
            'status': 'deleted',
            'updated_at': datetime.now().isoformat()
        }).eq('id', model_id).execute()
        
        logger.info(f"✅ Model marked as deleted")
        
        return JSONResponse({
            "success": True,
            "message": "Model deleted successfully",
            "model_id": model_id
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error deleting model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_user_model_stats(user_id: str):
    """
    Get statistics about user's models
    
    Args:
        user_id: User ID from authentication
    
    Returns:
        Statistics including total models, storage used, etc.
    """
    
    if not SUPABASE_AVAILABLE or not supabase:
        # Return empty stats if Supabase not available
        return JSONResponse({
            "success": True,
            "stats": {
                "user_id": user_id,
                "total_models": 0,
                "ready_models": 0,
                "training_models": 0,
                "failed_models": 0,
                "total_storage_bytes": 0,
                "total_storage_mb": 0,
                "last_model_created": None
            }
        })
    
    try:
        if not supabase:
            raise HTTPException(status_code=500, detail="Supabase not configured")
        
        logger.info(f"📊 Fetching stats for user {user_id}")
        
        # Use the view we created
        response = supabase.from_('user_model_stats').select('*').eq('user_id', user_id).single().execute()
        
        if response.data:
            stats = response.data
        else:
            # No models yet
            stats = {
                "user_id": user_id,
                "total_models": 0,
                "ready_models": 0,
                "training_models": 0,
                "failed_models": 0,
                "total_storage_bytes": 0,
                "last_model_created": None
            }
        
        # Convert bytes to human-readable
        storage_mb = stats['total_storage_bytes'] / (1024 * 1024) if stats['total_storage_bytes'] else 0
        
        return JSONResponse({
            "success": True,
            "stats": {
                **stats,
                "total_storage_mb": round(storage_mb, 2)
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Error fetching stats: {e}")
        # Return empty stats on error
        return JSONResponse({
            "success": True,
            "stats": {
                "user_id": user_id,
                "total_models": 0,
                "ready_models": 0,
                "training_models": 0,
                "failed_models": 0,
                "total_storage_bytes": 0,
                "total_storage_mb": 0,
                "last_model_created": None
            }
        })
