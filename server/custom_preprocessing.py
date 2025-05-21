from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pathlib import Path
import pandas as pd
import numpy as np
import json
import uuid
import os
from typing import List, Optional

# Create router for custom preprocessing
router = APIRouter(prefix="/custom-preprocessing")

# Create folder to store results if it doesn't exist
result_folder = Path("preprocessing_results")
result_folder.mkdir(exist_ok=True)

@router.post("/analyze-file/")
async def analyze_file(file: UploadFile = File(...)):
    """
    Analyze a file to get column information for custom preprocessing
    """
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
            
            # Get sample values
            sample_values = df[column].dropna().head(5).tolist()
            
            columns_info.append({
                "name": column,
                "current_type": current_type,
                "suggested_type": suggested_type,
                "sample_values": [str(x) for x in sample_values]
            })
        
        # Clean up temporary file
        os.remove(temp_file_path)
        
        return {
            "success": True,
            "filename": file.filename,
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns_info": columns_info
        }
        
    except Exception as e:
        # Clean up if needed
        if os.path.exists(temp_file_path):
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
                    if new_type == "int":
                        df[column] = pd.to_numeric(df[column], errors='coerce', downcast='integer')
                    elif new_type == "float":
                        df[column] = pd.to_numeric(df[column], errors='coerce', downcast='float')
                    elif new_type == "datetime":
                        df[column] = pd.to_datetime(df[column], errors='coerce')
                    else:  # string
                        df[column] = df[column].astype(str)
                    
                    transformation_report["transformations_applied"].append(f"Converted {column} to {new_type}")
        
        # Save transformed file
        output_filename = f"transformed_{uuid.uuid4()}_{file.filename}"
        output_path = result_folder / output_filename
        df.to_csv(output_path, index=False)
        
        # Save transformation report
        report_filename = f"{output_filename}.report.json"
        report_path = result_folder / report_filename
        
        with open(report_path, 'w') as f:
            json.dump(transformation_report, f, indent=2)
        
        # Clean up temporary file
        os.remove(temp_file_path)
        
        return {
            "success": True,
            "message": "Transformations applied successfully",
            "transformed_file": output_filename,
            "report_file": report_filename,
            "report": transformation_report
        }
        
    except Exception as e:
        # Clean up if needed
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        raise HTTPException(
            status_code=500,
            detail=f"Error applying transformations: {str(e)}"
        )

@router.post("/preview-transformation/")
async def preview_transformation(
    file: UploadFile = File(...),
    transformations: str = Form(...)
):
    """
    Preview how transformations would affect the data
    """
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
        
        # Create a copy for transformations
        transformed_df = df.copy()
        
        # Drop columns if specified
        if "columns_to_drop" in transform_config:
            columns_to_drop = transform_config["columns_to_drop"]
            if columns_to_drop:
                valid_columns_to_drop = [col for col in columns_to_drop if col in transformed_df.columns]
                if valid_columns_to_drop:
                    transformed_df = transformed_df.drop(columns=valid_columns_to_drop)
        
        # Apply data type transformations
        if "data_types" in transform_config:
            for column, new_type in transform_config["data_types"].items():
                if column in transformed_df.columns:
                    try:
                        if new_type == "int":
                            transformed_df[column] = pd.to_numeric(transformed_df[column], errors='coerce', downcast='integer')
                        elif new_type == "float":
                            transformed_df[column] = pd.to_numeric(transformed_df[column], errors='coerce', downcast='float')
                        elif new_type == "datetime":
                            transformed_df[column] = pd.to_datetime(transformed_df[column], errors='coerce')
                        else:  # string
                            transformed_df[column] = transformed_df[column].astype(str)
                    except Exception as e:
                        # If there's an error, just continue
                        pass
        
        # Prepare preview data (first 5 rows)
        preview_data = {
            "original": df.head(5).to_dict('records'),
            "transformed": transformed_df.head(5).to_dict('records'),
            "columns": {
                "original": df.columns.tolist(),
                "transformed": transformed_df.columns.tolist()
            }
        }
        
        # Clean up temporary file
        os.remove(temp_file_path)
        
        return {
            "success": True,
            "preview": preview_data
        }
        
    except Exception as e:
        # Clean up if needed
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        raise HTTPException(
            status_code=500,
            detail=f"Error previewing transformations: {str(e)}"
        )