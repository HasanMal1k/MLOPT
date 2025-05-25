from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pathlib import Path
import pandas as pd
import numpy as np
import json
import uuid
import os
from typing import List, Optional
import math

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

@router.post("/apply-transformations/")
async def apply_transformations(
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
        
        # Track changes for reporting
        transformation_report = {
            "data_types": {},
            "columns_dropped": [],
            "transformations_applied": []
        }
        
        # Drop columns first
        if "columns_to_drop" in transform_config:
            columns_to_drop = transform_config["columns_to_drop"]
            if columns_to_drop:
                # Filter to only include columns that exist in the dataframe
                valid_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
                if valid_columns_to_drop:
                    df = df.drop(columns=valid_columns_to_drop)
                    transformation_report["columns_dropped"] = valid_columns_to_drop
                    for col in valid_columns_to_drop:
                        transformation_report["transformations_applied"].append(f"Dropped column {col}")
        
        # Apply data type transformations
        if "data_types" in transform_config:
            for column, new_type in transform_config["data_types"].items():
                if column in df.columns:
                    # Store original data type
                    transformation_report["data_types"][column] = {
                        "original": str(df[column].dtype),
                        "converted_to": new_type
                    }
                    
                    # Convert data type
                    try:
                        if new_type == "int":
                            df[column] = pd.to_numeric(df[column], errors='coerce', downcast='integer')
                        elif new_type == "float":
                            df[column] = pd.to_numeric(df[column], errors='coerce', downcast='float')
                        elif new_type == "datetime":
                            df[column] = pd.to_datetime(df[column], errors='coerce')
                        else:  # string
                            df[column] = df[column].astype(str)
                        
                        transformation_report["transformations_applied"].append(f"Converted {column} to {new_type}")
                    except Exception as conv_error:
                        transformation_report["transformations_applied"].append(f"Failed to convert {column} to {new_type}: {str(conv_error)}")
        
        # Clean DataFrame of any remaining NaN/inf values before saving
        # Replace inf with NaN, then handle NaN appropriately
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # For numeric columns, fill NaN with 0 or median
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isna().any():
                # Use median if available, otherwise 0
                fill_value = df[col].median() if not df[col].isna().all() else 0
                if pd.isna(fill_value):
                    fill_value = 0
                df[col] = df[col].fillna(fill_value)
        
        # For non-numeric columns, fill NaN with empty string
        for col in df.select_dtypes(exclude=[np.number]).columns:
            df[col] = df[col].fillna('')
        
        # Save transformed file
        output_filename = f"transformed_{uuid.uuid4()}_{file.filename}"
        output_path = result_folder / output_filename
        df.to_csv(output_path, index=False)
        
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

@router.post("/preview-transformations/")
async def preview_transformations(
    file: UploadFile = File(...),
    transformations: str = Form(...)
):
    """Preview transformations before applying them"""
    try:
        # Parse the transformations
        transform_config = json.loads(transformations)
        
        # Generate a unique filename
        unique_id = str(uuid.uuid4())[:8]
        original_filename = file.filename
        safe_filename = f"{unique_id}_{original_filename}"
        
        # Save the uploaded file temporarily
        temp_file_path = Path(f"temp_{safe_filename}")
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
        os.remove(temp_file_path)
        
        # Apply transformations to get preview
        transformed_df, applied_transforms = apply_transformations(df.copy(), transform_config)
        
        # Get first 5 rows for preview
        original_preview = df.head(5).to_dict('records')
        transformed_preview = transformed_df.head(5).to_dict('records')
        
        # Clean the data for JSON serialization
        def clean_for_json(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                if pd.isna(obj):
                    return None
                return float(obj)
            elif pd.isna(obj):
                return None
            else:
                return obj
        
        # Clean the preview data
        original_clean = []
        for row in original_preview:
            clean_row = {k: clean_for_json(v) for k, v in row.items()}
            original_clean.append(clean_row)
            
        transformed_clean = []
        for row in transformed_preview:
            clean_row = {k: clean_for_json(v) for k, v in row.items()}
            transformed_clean.append(clean_row)
        
        return {
            "success": True,
            "original": original_clean,
            "transformed": transformed_clean,
            "columns": {
                "original": list(df.columns),
                "transformed": list(transformed_df.columns)
            },
            "transformations_applied": applied_transforms
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )