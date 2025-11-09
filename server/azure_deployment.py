"""
Azure ML Deployment API - Deploy models to Azure Machine Learning
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/deployments", tags=["Azure Deployments"])

# Try to import Azure and MLflow dependencies
try:
    from azure.ai.ml import MLClient
    from azure.identity import ClientSecretCredential
    from azure.ai.ml.entities import Model, ManagedOnlineDeployment, ManagedOnlineEndpoint
    from azure.ai.ml.constants import AssetTypes
    import mlflow
    import mlflow.sklearn
    import joblib
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    logger.warning("⚠️ Azure ML or MLflow modules not installed. Deployment features will be disabled.")
    logger.warning("   To enable: pip install azure-ai-ml azure-identity mlflow")

# Try to import Supabase
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    logger.warning("⚠️ Supabase module not installed")

# Supabase client
supabase = None
if SUPABASE_AVAILABLE:
    SUPABASE_URL = os.getenv("SUPABASE_URL", "")
    SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
    
    if SUPABASE_URL and SUPABASE_SERVICE_KEY:
        try:
            supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
            logger.info("✅ Supabase client initialized for deployments")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Supabase: {e}")

# Azure credentials from environment
AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID", "")
AZURE_CLIENT_ID = os.getenv("AZURE_CLIENT_ID", "")
AZURE_CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET", "")
AZURE_SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID", "")
AZURE_RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP", "")
AZURE_WORKSPACE_NAME = os.getenv("AZURE_WORKSPACE_NAME", "")

# Pydantic models
class DeployModelRequest(BaseModel):
    model_id: str
    endpoint_name: Optional[str] = None  # If None, create new endpoint
    instance_type: str = "Standard_DS1_v2"
    instance_count: int = 1
    description: Optional[str] = None

class DeploymentResponse(BaseModel):
    deployment_id: str
    status: str
    message: str
    endpoint_name: Optional[str] = None
    scoring_uri: Optional[str] = None

@router.post("/deploy")
async def deploy_model(
    request: DeployModelRequest,
    background_tasks: BackgroundTasks,
    user_id: str
):
    """
    Deploy a trained model to Azure ML
    """
    if not AZURE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Azure ML deployment is not configured. Install azure-ai-ml and mlflow packages."
        )
    
    if not supabase:
        raise HTTPException(
            status_code=503,
            detail="Database not configured"
        )
    
    # Verify Azure credentials
    if not all([AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, 
                AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_WORKSPACE_NAME]):
        raise HTTPException(
            status_code=503,
            detail="Azure credentials not configured. Set AZURE_* environment variables."
        )
    
    try:
        # Get model from database
        response = supabase.table("trained_models")\
            .select("*")\
            .eq("id", request.model_id)\
            .eq("user_id", user_id)\
            .execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Model not found")
        
        model_data = response.data[0]
        
        # Generate deployment name with microseconds for uniqueness
        now = datetime.now()
        timestamp = now.strftime('%Y%m%d%H%M%S')
        timestamp_ms = now.strftime('%Y%m%d%H%M%S%f')  # Include microseconds for unique model name
        deployment_name = f"deploy-{timestamp}"
        
        # Reuse endpoint if provided, otherwise create user-specific shared endpoint
        if request.endpoint_name:
            endpoint_name = request.endpoint_name
        else:
            # Use a shared endpoint per user for faster deployments
            endpoint_name = f"mlopt-{user_id[:8]}-endpoint"
        
        azure_model_name = f"model_{model_data['model_name'].replace(' ', '_')}_{timestamp_ms}"
        
        # Create deployment record in database
        deployment_record = {
            "user_id": user_id,
            "model_id": request.model_id,
            "deployment_name": deployment_name,
            "endpoint_name": endpoint_name,
            "azure_model_name": azure_model_name,
            "instance_type": request.instance_type,
            "instance_count": request.instance_count,
            "status": "deploying",
            "description": request.description,
            "deployment_config": {
                "algorithm": model_data.get("algorithm"),
                "model_type": model_data.get("model_type")
            }
        }
        
        result = supabase.table("deployments").insert(deployment_record).execute()
        deployment_id = result.data[0]["id"]
        
        logger.info(f"🚀 Starting deployment {deployment_id} for user {user_id}")
        
        # Start deployment in background
        background_tasks.add_task(
            deploy_to_azure,
            deployment_id=deployment_id,
            model_data=model_data,
            azure_model_name=azure_model_name,
            deployment_name=deployment_name,
            endpoint_name=endpoint_name,
            instance_type=request.instance_type,
            instance_count=request.instance_count
        )
        
        return JSONResponse({
            "deployment_id": deployment_id,
            "status": "deploying",
            "message": "Deployment started. This may take 5-10 minutes.",
            "endpoint_name": endpoint_name
        })
        
    except Exception as e:
        logger.error(f"Deployment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def deploy_to_azure(
    deployment_id: str,
    model_data: Dict,
    azure_model_name: str,
    deployment_name: str,
    endpoint_name: str,
    instance_type: str,
    instance_count: int
):
    """
    Background task to deploy model to Azure ML
    """
    try:
        logger.info(f"📦 Step 1: Loading model file for deployment {deployment_id}")
        
        # Download model file from Supabase Storage
        model_file_path = model_data["model_file_path"]
        local_model_path = Path("temp_models") / f"{deployment_id}.pkl"
        local_model_path.parent.mkdir(exist_ok=True)
        
        # Download from Supabase
        file_data = supabase.storage.from_("model-files").download(model_file_path)
        with open(local_model_path, "wb") as f:
            f.write(file_data)
        
        logger.info(f"✅ Model downloaded to {local_model_path}")
        
        # Load and convert to MLflow format
        logger.info(f"🔄 Step 2: Converting to MLflow format")
        model = joblib.load(local_model_path)
        mlflow_model_path = Path("temp_models") / f"mlflow_{deployment_id}"
        
        # Save as MLflow model (sklearn format)
        mlflow.sklearn.save_model(sk_model=model, path=str(mlflow_model_path))
        logger.info(f"✅ MLflow model saved")
        
        # Authenticate with Azure
        logger.info(f"🔐 Step 3: Authenticating with Azure")
        credential = ClientSecretCredential(
            tenant_id=AZURE_TENANT_ID,
            client_id=AZURE_CLIENT_ID,
            client_secret=AZURE_CLIENT_SECRET
        )
        
        ml_client = MLClient(
            credential,
            AZURE_SUBSCRIPTION_ID,
            AZURE_RESOURCE_GROUP,
            AZURE_WORKSPACE_NAME
        )
        logger.info("✅ Azure authenticated")
        
        # Register model with retry logic for version conflicts
        logger.info(f"📝 Step 4: Registering model to Azure ML")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # If retry, append attempt number to model name
                current_model_name = azure_model_name if attempt == 0 else f"{azure_model_name}_v{attempt + 1}"
                
                model_asset = Model(
                    path=str(mlflow_model_path),
                    type=AssetTypes.MLFLOW_MODEL,
                    name=current_model_name,
                    description=model_data.get("description", "Auto-deployed model")
                )
                
                registered_model = ml_client.models.create_or_update(model_asset)
                azure_model_name = current_model_name  # Update name if we used retry variant
                break
            except Exception as model_error:
                if "ModelVersionInUse" in str(model_error) or "already exists" in str(model_error):
                    if attempt < max_retries - 1:
                        logger.warning(f"⚠️ Model name collision, retrying with variant name (attempt {attempt + 2}/{max_retries})")
                        continue
                    else:
                        logger.error(f"❌ Failed to register model after {max_retries} attempts")
                        raise
                else:
                    # Non-conflict error, raise immediately
                    raise
        logger.info(f"✅ Model registered: {registered_model.name} v{registered_model.version}")
        
        # Create or get endpoint
        logger.info(f"🌐 Step 5: Setting up endpoint")
        try:
            endpoint = ml_client.online_endpoints.get(name=endpoint_name)
            logger.info(f"✅ Using existing endpoint: {endpoint_name}")
        except:
            logger.info(f"Creating new endpoint: {endpoint_name}")
            endpoint = ManagedOnlineEndpoint(
                name=endpoint_name,
                description=f"Endpoint for {model_data['model_name']}",
                auth_mode="key"
            )
            endpoint = ml_client.online_endpoints.begin_create_or_update(endpoint).result()
            logger.info(f"✅ Endpoint created: {endpoint_name}")
        
        # Create deployment
        logger.info(f"🚀 Step 6: Creating deployment")
        deployment = ManagedOnlineDeployment(
            name=deployment_name,
            endpoint_name=endpoint_name,
            model=registered_model.id,
            instance_type=instance_type,
            instance_count=instance_count,
        )
        
        deployment_result = ml_client.online_deployments.begin_create_or_update(deployment).result()
        logger.info(f"✅ Deployment created: {deployment_name}")
        
        # Get final endpoint details
        endpoint = ml_client.online_endpoints.get(name=endpoint_name)
        
        # Build swagger URI from scoring URI (Azure ML pattern)
        swagger_uri = None
        if endpoint.scoring_uri:
            # Convert scoring URI to swagger URI
            # Pattern: https://<endpoint>.eastus2.inference.ml.azure.com/score -> /swagger.json
            swagger_uri = endpoint.scoring_uri.replace('/score', '/swagger.json')
        
        # Update deployment record
        supabase.table("deployments").update({
            "status": "active",
            "azure_model_version": str(registered_model.version),
            "scoring_uri": endpoint.scoring_uri,
            "swagger_uri": swagger_uri,
            "deployed_at": datetime.now().isoformat()
        }).eq("id", deployment_id).execute()
        
        logger.info(f"🎉 Deployment {deployment_id} completed successfully!")
        logger.info(f"   Scoring URI: {endpoint.scoring_uri}")
        
        # Cleanup temp files
        local_model_path.unlink(missing_ok=True)
        
    except Exception as e:
        logger.error(f"❌ Deployment {deployment_id} failed: {e}")
        import traceback
        error_trace = traceback.format_exc()
        logger.error(error_trace)
        
        # Update status to failed
        if supabase:
            supabase.table("deployments").update({
                "status": "failed",
                "error_message": str(e)
            }).eq("id", deployment_id).execute()

@router.get("/list")
async def list_deployments(user_id: str, status: Optional[str] = None):
    """List all deployments for a user"""
    if not supabase:
        raise HTTPException(status_code=503, detail="Database not configured")
    
    try:
        query = supabase.table("deployments")\
            .select("*, trained_models(model_name, algorithm, model_type)")\
            .eq("user_id", user_id)\
            .neq("status", "deleted")\
            .order("created_at", desc=True)
        
        if status:
            query = query.eq("status", status)
        
        response = query.execute()
        
        return JSONResponse({
            "deployments": response.data,
            "total": len(response.data)
        })
    except Exception as e:
        logger.error(f"Error listing deployments: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_deployment_stats(user_id: str):
    """Get deployment statistics for a user"""
    if not supabase:
        return JSONResponse({"total": 0, "active": 0, "deploying": 0, "failed": 0})
    
    try:
        response = supabase.rpc("user_deployment_stats")\
            .eq("user_id", user_id)\
            .execute()
        
        if response.data:
            return JSONResponse(response.data[0])
        return JSONResponse({"total": 0, "active": 0, "deploying": 0, "failed": 0})
    except:
        return JSONResponse({"total": 0, "active": 0, "deploying": 0, "failed": 0})

@router.delete("/{deployment_id}")
async def delete_deployment(deployment_id: str, user_id: str):
    """Soft delete a deployment"""
    if not supabase:
        raise HTTPException(status_code=503, detail="Database not configured")
    
    try:
        supabase.table("deployments").update({
            "status": "deleted",
            "deleted_at": datetime.now().isoformat()
        }).eq("id", deployment_id).eq("user_id", user_id).execute()
        
        return JSONResponse({"message": "Deployment deleted successfully"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
