from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pathlib import Path
import pandas as pd
import numpy as np
import json
import uuid
import os
import logging
from typing import List, Optional
import math

# Import the proper apply_transformations function from transformations module
from transformations import apply_transformations as apply_specific_transformations

# Setup logger
logger = logging.getLogger(__name__)

# Create router for custom preprocessing
router = APIRouter(prefix="/custom-preprocessing")

# Create folder to store results if it doesn't exist
result_folder = Path("preprocessing_results")
result_folder.mkdir(exist_ok=True)

def clean_for_json(obj):
    """
    Recursively clean an object to make it JSON serializable by replacing NaN/inf values
    """
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(clean_for_json(item) for item in obj)
    elif isinstance(obj, (np.ndarray, pd.Series)):
        return clean_for_json(obj.tolist())
    elif isinstance(obj, pd.DataFrame):
        return clean_for_json(obj.to_dict('records'))
    elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        if pd.isna(obj) or math.isnan(float(obj)):
            return None
        elif math.isinf(float(obj)):
            return None  # or return a string like "infinity"
        else:
            return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (float, int)):
        if pd.isna(obj) or (isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj))):
            return None
        return obj
    elif pd.isna(obj):
        return None
    else:
        return obj

def apply_transformations(df: pd.DataFrame, transform_config: dict) -> tuple[pd.DataFrame, list]:
    """
    Apply transformations to a dataframe and return the result plus applied transforms
    This function now uses the transformations module for all transformation types
    """
    # Use the proper function from transformations.py that handles all transformation types
    df_transformed, applied_transforms = apply_specific_transformations(df, transform_config)
    
    # Additionally handle columns_to_drop and data_types if present
    # (these are custom preprocessing specific, not in transformations.py)
    
    # Drop columns if specified
    if "columns_to_drop" in transform_config:
        columns_to_drop = transform_config["columns_to_drop"]
        if columns_to_drop:
            # Filter to only include columns that exist in the dataframe
            valid_columns_to_drop = [col for col in columns_to_drop if col in df_transformed.columns]
            if valid_columns_to_drop:
                df_transformed = df_transformed.drop(columns=valid_columns_to_drop)
                applied_transforms.append(f"Dropped columns: {', '.join(valid_columns_to_drop)}")
    
    # Apply data type transformations
    if "data_types" in transform_config:
        for column, new_type in transform_config["data_types"].items():
            if column in df_transformed.columns:
                try:
                    if new_type == "int":
                        df_transformed[column] = pd.to_numeric(df_transformed[column], errors='coerce', downcast='integer')
                    elif new_type == "float":
                        df_transformed[column] = pd.to_numeric(df_transformed[column], errors='coerce', downcast='float')
                    elif new_type == "datetime":
                        df_transformed[column] = pd.to_datetime(df_transformed[column], errors='coerce')
                    else:  # string
                        df_transformed[column] = df_transformed[column].astype(str)
                    
                    applied_transforms.append(f"Converted {column} to {new_type}")
                except Exception as conv_error:
                    applied_transforms.append(f"Failed to convert {column} to {new_type}: {str(conv_error)}")
    
    return df_transformed, applied_transforms

def clean_preview_data(data):
    """Clean preview data for JSON serialization"""
    cleaned = []
    for row in data:
        clean_row = {}
        for key, value in row.items():
            if pd.isna(value):
                clean_row[key] = None
            elif isinstance(value, (np.integer, np.int64)):
                clean_row[key] = int(value)
            elif isinstance(value, (np.floating, np.float64)):
                if pd.isna(value) or np.isnan(value) or np.isinf(value):
                    clean_row[key] = None
                else:
                    clean_row[key] = float(value)
            else:
                clean_row[key] = str(value) if value is not None else None
        cleaned.append(clean_row)
    return cleaned

@router.post("/analyze-file/")
async def analyze_file(file: UploadFile = File(...)):
    """
    Analyze a file to get column information for custom preprocessing
    """
    temp_file_path = None
    try:
        # Save the file temporarily
        file_content = await file.read()
        temp_file_path = f"temp_{uuid.uuid4()}.csv"
        
        with open(temp_file_path, "wb") as f:
            f.write(file_content)
        
        # Read the file with pandas
        df = pd.read_csv(temp_file_path)
        
        # Get column information
        columns_info = []
        for column in df.columns:
            # Get current data type
            current_type = str(df[column].dtype)
            
            # Get suggested type based on current type
            suggested_type = "string"  # default
            if current_type in ["int64", "int32"]:
                suggested_type = "int"
            elif current_type in ["float64", "float32"]:
                suggested_type = "float"
            elif "datetime" in current_type:
                suggested_type = "datetime"
            
            # Get sample values - clean them for JSON
            sample_values = df[column].dropna().head(5).tolist()
            # Clean sample values to remove NaN/inf
            clean_sample_values = []
            for val in sample_values:
                if pd.isna(val) or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
                    clean_sample_values.append(None)
                else:
                    clean_sample_values.append(str(val))
            
            columns_info.append({
                "name": column,
                "current_type": current_type,
                "suggested_type": suggested_type,
                "sample_values": clean_sample_values
            })
        
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        response_data = {
            "success": True,
            "filename": file.filename,
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns_info": columns_info
        }
        
        # Clean the entire response for JSON serialization
        clean_response = clean_for_json(response_data)
        
        return JSONResponse(content=clean_response)
        
    except Exception as e:
        # Clean up if needed
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing file: {str(e)}"
        )

@router.post("/preview-transformation/")
async def preview_transformation(
    file: UploadFile = File(...),
    transformations: str = Form(...)
):
    """Preview transformations before applying them - FIXED ENDPOINT NAME"""
    temp_file_path = None
    try:
        # Parse the transformations
        transform_config = json.loads(transformations)
        
        # Generate a unique filename
        unique_id = str(uuid.uuid4())[:8]
        original_filename = file.filename
        safe_filename = f"{unique_id}_{original_filename}"
        
        # Save the uploaded file temporarily
        temp_file_path = f"temp_{safe_filename}"
        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Read the file
        if original_filename.lower().endswith('.csv'):
            df = pd.read_csv(temp_file_path)
        elif original_filename.lower().endswith('.xlsx'):
            df = pd.read_excel(temp_file_path)
        else:
            raise HTTPException(status_code=400, detail="File format not supported")
        
        # Remove the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        # Apply transformations to get preview
        transformed_df, applied_transforms = apply_transformations(df.copy(), transform_config)
        
        # Get first 5 rows for preview
        original_preview = df.head(5).to_dict('records')
        transformed_preview = transformed_df.head(5).to_dict('records')
        
        # Clean the preview data
        original_clean = clean_preview_data(original_preview)
        transformed_clean = clean_preview_data(transformed_preview)
        
        return JSONResponse(content={
            "success": True,
            "preview": {
                "original": original_clean,
                "transformed": transformed_clean,
                "columns": {
                    "original": list(df.columns),
                    "transformed": list(transformed_df.columns)
                }
            },
            "transformations_applied": applied_transforms
        })
    
    except Exception as e:
        # Clean up if needed
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@router.post("/apply-transformations/")
async def apply_transformations_endpoint(
    file: UploadFile = File(...),
    transformations: str = Form(...)
):
    """
    Apply custom transformations to a file based on user selections
    """
    temp_file_path = None
    try:
        # Parse transformations
        transform_config = json.loads(transformations)
        
        # Save the file temporarily
        file_content = await file.read()
        temp_file_path = f"temp_{uuid.uuid4()}.csv"
        
        with open(temp_file_path, "wb") as f:
            f.write(file_content)
        
        # Read the file with pandas
        df = pd.read_csv(temp_file_path)
        
        # Apply transformations
        transformed_df, applied_transforms = apply_transformations(df, transform_config)
        
        # Track changes for reporting
        transformation_report = {
            "data_types": {},
            "columns_dropped": [],
            "transformations_applied": applied_transforms
        }
        
        # Record what was actually done
        if "columns_to_drop" in transform_config:
            columns_to_drop = transform_config["columns_to_drop"]
            if columns_to_drop:
                valid_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
                transformation_report["columns_dropped"] = valid_columns_to_drop
        
        if "data_types" in transform_config:
            for column, new_type in transform_config["data_types"].items():
                if column in df.columns and column in transformed_df.columns:
                    transformation_report["data_types"][column] = {
                        "original": str(df[column].dtype),
                        "converted_to": new_type
                    }
        
        # Clean DataFrame of any remaining NaN/inf values before saving
        # Replace inf with NaN, then handle NaN appropriately
        transformed_df = transformed_df.replace([np.inf, -np.inf], np.nan)
        
        # For numeric columns, fill NaN with 0 or median
        for col in transformed_df.select_dtypes(include=[np.number]).columns:
            if transformed_df[col].isna().any():
                # Use median if available, otherwise 0
                fill_value = transformed_df[col].median() if not transformed_df[col].isna().all() else 0
                if pd.isna(fill_value):
                    fill_value = 0
                transformed_df[col] = transformed_df[col].fillna(fill_value)
        
        # For non-numeric columns, fill NaN with empty string
        for col in transformed_df.select_dtypes(exclude=[np.number]).columns:
            transformed_df[col] = transformed_df[col].fillna('')
        
        # Save transformed file
        output_filename = f"transformed_{uuid.uuid4()}_{file.filename}"
        output_path = result_folder / output_filename
        transformed_df.to_csv(output_path, index=False)
        
        # Save transformation pipeline for reproducible predictions
        try:
            import joblib
            pipeline_path = result_folder / f"{output_filename.replace('.csv', '')}_pipeline.joblib"
            
            transformation_pipeline = {
                "pipeline_type": "custom_transformations",
                "original_columns": list(df.columns),
                "final_columns": list(transformed_df.columns),
                "transform_config": transform_config,
                "transformations_applied": applied_transforms,
                "original_shape": list(df.shape),
                "final_shape": list(transformed_df.shape)
            }
            
            joblib.dump(transformation_pipeline, pipeline_path)
            logger.info(f"Saved custom transformation pipeline to {pipeline_path}")
            
        except Exception as e:
            logger.warning(f"Could not save transformation pipeline: {e}")
        
        # Save transformation report
        report_filename = f"{output_filename}.report.json"
        report_path = result_folder / report_filename
        
        # Clean the report before saving
        clean_report = clean_for_json(transformation_report)
        
        with open(report_path, 'w') as f:
            json.dump(clean_report, f, indent=2)
        
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        response_data = {
            "success": True,
            "message": "Transformations applied successfully",
            "transformed_file": output_filename,
            "report_file": report_filename,
            "report": clean_report
        }
        
        # Clean the entire response for JSON serialization
        clean_response = clean_for_json(response_data)
        
        return JSONResponse(content=clean_response)
        
    except Exception as e:
        # Clean up if needed
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        raise HTTPException(
            status_code=500,
            detail=f"Error applying transformations: {str(e)}"
        )

# Keep the old endpoint for backward compatibility (with 's')
@router.post("/preview-transformations/")
async def preview_transformations_legacy(
    file: UploadFile = File(...),
    transformations: str = Form(...)
):
    """Legacy endpoint - redirects to the correct one"""
    return await preview_transformation(file, transformations)