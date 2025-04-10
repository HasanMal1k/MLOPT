from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pathlib import Path
import pandas as pd
import numpy as np
import json
import uuid
import time
from typing import List, Optional
import os

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
            
            # Detect possible data types
            possible_types = ["string", "integer", "float", "datetime", "category"]
            
            # Check if column can be converted to numeric
            is_numeric = pd.to_numeric(df[column], errors='coerce').notna().all()
            is_datetime = pd.to_datetime(df[column], errors='coerce').notna().all()
            
            # Auto-suggest best data type
            suggested_type = "string"  # default
            if current_type in ["int64", "int32"]:
                suggested_type = "integer"
            elif current_type in ["float64", "float32"]:
                suggested_type = "float"
            elif "datetime" in current_type:
                suggested_type = "datetime"
            elif is_datetime:
                suggested_type = "datetime"
            elif is_numeric:
                # Check if it's integer-like
                if pd.to_numeric(df[column], errors='coerce').dropna().apply(lambda x: x.is_integer()).all():
                    suggested_type = "integer"
                else:
                    suggested_type = "float"
            
            # Get sample values
            sample_values = df[column].dropna().head(5).tolist()
            
            # Calculate missing value percentage
            missing_percentage = (df[column].isna().sum() / len(df)) * 100
            
            # Get unique values count and percentage
            unique_count = df[column].nunique()
            unique_percentage = (unique_count / len(df)) * 100
            
            # For categorical columns, get value distribution
            value_distribution = {}
            if unique_count <= 20:  # Only for columns with reasonable number of categories
                value_counts = df[column].value_counts(normalize=True)
                value_distribution = {str(k): float(v) for k, v in value_counts.head(10).items()}
            
            columns_info.append({
                "name": column,
                "current_type": current_type,
                "suggested_type": suggested_type,
                "possible_types": possible_types,
                "sample_values": [str(x) for x in sample_values],
                "missing_percentage": round(missing_percentage, 2),
                "unique_count": int(unique_count),
                "unique_percentage": round(unique_percentage, 2),
                "value_distribution": value_distribution
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
    transformations: str = Form(...),
    background_tasks: BackgroundTasks = None
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
            "column_changes": {},
            "row_count": {
                "before": len(df),
                "after": None
            },
            "data_types": {},
            "missing_values": {},
            "transformations_applied": []
        }
        
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
                    if new_type == "integer":
                        df[column] = pd.to_numeric(df[column], errors='coerce', downcast='integer')
                    elif new_type == "float":
                        df[column] = pd.to_numeric(df[column], errors='coerce', downcast='float')
                    elif new_type == "datetime":
                        df[column] = pd.to_datetime(df[column], errors='coerce')
                    elif new_type == "category":
                        df[column] = df[column].astype('category')
                    else:  # string
                        df[column] = df[column].astype(str)
                    
                    transformation_report["transformations_applied"].append(f"Converted {column} to {new_type}")
        
        # Handle missing values
        if "missing_values" in transform_config:
            for column, action in transform_config["missing_values"].items():
                if column in df.columns:
                    missing_before = df[column].isna().sum()
                    
                    if action["method"] == "drop":
                        # Drop rows with missing values in this column
                        df = df.dropna(subset=[column])
                        transformation_report["transformations_applied"].append(f"Dropped rows with missing values in {column}")
                    
                    elif action["method"] == "fill_value":
                        # Fill with a specific value
                        fill_value = action.get("value", "")
                        df[column] = df[column].fillna(fill_value)
                        transformation_report["transformations_applied"].append(f"Filled missing values in {column} with '{fill_value}'")
                    
                    elif action["method"] == "mean":
                        # Fill with mean (for numeric columns)
                        df[column] = df[column].fillna(df[column].mean())
                        transformation_report["transformations_applied"].append(f"Filled missing values in {column} with column mean")
                    
                    elif action["method"] == "median":
                        # Fill with median (for numeric columns)
                        df[column] = df[column].fillna(df[column].median())
                        transformation_report["transformations_applied"].append(f"Filled missing values in {column} with column median")
                    
                    missing_after = df[column].isna().sum()
                    transformation_report["missing_values"][column] = {
                        "before": int(missing_before),
                        "after": int(missing_after),
                        "method": action["method"]
                    }
        
        # Handle column operations
        if "column_operations" in transform_config:
            for operation in transform_config["column_operations"]:
                if operation["type"] == "drop" and operation["column"] in df.columns:
                    df = df.drop(columns=[operation["column"]])
                    transformation_report["transformations_applied"].append(f"Dropped column {operation['column']}")
                
                elif operation["type"] == "rename" and operation["old_name"] in df.columns:
                    df = df.rename(columns={operation["old_name"]: operation["new_name"]})
                    transformation_report["transformations_applied"].append(f"Renamed column {operation['old_name']} to {operation['new_name']}")
                
                elif operation["type"] == "create":
                    # Basic column creation using a formula - this is simplified
                    # A real implementation would need to evaluate expressions safely
                    if operation["formula"] == "concat":
                        col1 = operation["columns"][0]
                        col2 = operation["columns"][1]
                        df[operation["name"]] = df[col1].astype(str) + df[col2].astype(str)
                        transformation_report["transformations_applied"].append(f"Created new column {operation['name']} by concatenating {col1} and {col2}")
        
        # Update row count after transformations
        transformation_report["row_count"]["after"] = len(df)
        
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

@router.post("/feature-engineering/")
async def feature_engineering(
    file: UploadFile = File(...),
    operations: str = Form(...)
):
    """
    Apply feature engineering operations to create new columns
    """
    try:
        # Parse operations
        operations_list = json.loads(operations)
        
        # Save the file temporarily
        file_content = await file.read()
        temp_file_path = f"temp_{uuid.uuid4()}.csv"
        
        with open(temp_file_path, "wb") as f:
            f.write(file_content)
        
        # Read the file with pandas
        df = pd.read_csv(temp_file_path)
        
        # Track changes
        engineering_report = {
            "original_columns": list(df.columns),
            "new_columns": [],
            "operations_applied": []
        }
        
        # Apply operations
        for operation in operations_list:
            if operation["type"] == "datetime_features" and operation["column"] in df.columns:
                # Convert to datetime if not already
                df[operation["column"]] = pd.to_datetime(df[operation["column"]], errors='coerce')
                
                # Extract features
                for feature in operation["features"]:
                    if feature == "year":
                        new_col = f"{operation['column']}_year"
                        df[new_col] = df[operation["column"]].dt.year
                        engineering_report["new_columns"].append(new_col)
                        engineering_report["operations_applied"].append(f"Extracted year from {operation['column']}")
                    
                    elif feature == "month":
                        new_col = f"{operation['column']}_month"
                        df[new_col] = df[operation["column"]].dt.month
                        engineering_report["new_columns"].append(new_col)
                        engineering_report["operations_applied"].append(f"Extracted month from {operation['column']}")
                    
                    elif feature == "day":
                        new_col = f"{operation['column']}_day"
                        df[new_col] = df[operation["column"]].dt.day
                        engineering_report["new_columns"].append(new_col)
                        engineering_report["operations_applied"].append(f"Extracted day from {operation['column']}")
                    
                    elif feature == "dayofweek":
                        new_col = f"{operation['column']}_dayofweek"
                        df[new_col] = df[operation["column"]].dt.dayofweek
                        engineering_report["new_columns"].append(new_col)
                        engineering_report["operations_applied"].append(f"Extracted day of week from {operation['column']}")
            
            elif operation["type"] == "one_hot_encoding" and operation["column"] in df.columns:
                # Get column
                col = df[operation["column"]]
                
                # Apply one-hot encoding
                dummies = pd.get_dummies(col, prefix=operation["column"], drop_first=operation.get("drop_first", False))
                
                # Add to dataframe
                df = pd.concat([df, dummies], axis=1)
                
                # Update report
                new_cols = list(dummies.columns)
                engineering_report["new_columns"].extend(new_cols)
                engineering_report["operations_applied"].append(f"Applied one-hot encoding to {operation['column']}, created {len(new_cols)} new columns")
            
            elif operation["type"] == "binning" and operation["column"] in df.columns:
                # Get column
                col = df[operation["column"]]
                
                # Create bins
                bins = operation.get("bins", 3)
                bin_labels = operation.get("labels", [f"bin_{i}" for i in range(bins)])
                
                # Apply binning
                new_col = f"{operation['column']}_binned"
                df[new_col] = pd.cut(col, bins=bins, labels=bin_labels)
                
                # Update report
                engineering_report["new_columns"].append(new_col)
                engineering_report["operations_applied"].append(f"Applied binning to {operation['column']}, created {bins} bins")
        
        # Save transformed file
        output_filename = f"featured_{uuid.uuid4()}_{file.filename}"
        output_path = result_folder / output_filename
        df.to_csv(output_path, index=False)
        
        # Save report
        report_filename = f"{output_filename}.report.json"
        report_path = result_folder / report_filename
        
        with open(report_path, 'w') as f:
            json.dump(engineering_report, f, indent=2)
        
        # Clean up temporary file
        os.remove(temp_file_path)
        
        return {
            "success": True,
            "message": "Feature engineering applied successfully",
            "transformed_file": output_filename,
            "original_column_count": len(engineering_report["original_columns"]),
            "new_column_count": len(engineering_report["new_columns"]),
            "total_column_count": len(df.columns),
            "report": engineering_report
        }
        
    except Exception as e:
        # Clean up if needed
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        raise HTTPException(
            status_code=500,
            detail=f"Error applying feature engineering: {str(e)}"
        )

@router.get("/available-transformations/")
async def get_available_transformations():
    """
    Get a list of all available transformations for custom preprocessing
    """
    return {
        "data_type_transformations": [
            {"id": "string", "name": "Text (String)", "description": "Convert to text data"},
            {"id": "integer", "name": "Integer Number", "description": "Convert to whole numbers"},
            {"id": "float", "name": "Decimal Number", "description": "Convert to numbers with decimals"},
            {"id": "datetime", "name": "Date & Time", "description": "Convert to date/time format"},
            {"id": "category", "name": "Category", "description": "Convert to categorical data"}
        ],
        "missing_value_methods": [
            {"id": "drop", "name": "Drop Rows", "description": "Remove rows with missing values"},
            {"id": "fill_value", "name": "Fill with Value", "description": "Replace with a specific value"},
            {"id": "mean", "name": "Fill with Mean", "description": "Replace with column average (numeric)"},
            {"id": "median", "name": "Fill with Median", "description": "Replace with column median (numeric)"}
        ],
        "feature_engineering_options": [
            {"id": "datetime_features", "name": "Date/Time Features", "description": "Extract components from date columns"},
            {"id": "one_hot_encoding", "name": "One-Hot Encoding", "description": "Convert categories to binary columns"},
            {"id": "binning", "name": "Numeric Binning", "description": "Group numeric values into categories"}
        ]
    }