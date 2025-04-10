from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
import polars as pl
import pandas as pd
import numpy as np
import joblib
import re
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import os
import time
import json
import uuid

# Import custom preprocessing module
from custom_preprocessing import router as custom_preprocessing_router

app = FastAPI()

# Register custom preprocessing router
app.include_router(custom_preprocessing_router)

origins = [
    "http://localhost:3000",  # Next.js app URL
    "http://127.0.0.1:3000",
    "https://your-next-app-domain.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create folders to store uploaded and processed files
upload_folder = Path("files")
processed_folder = Path("processed_files")
status_folder = Path("processing_status")
upload_folder.mkdir(exist_ok=True)
processed_folder.mkdir(exist_ok=True)
status_folder.mkdir(exist_ok=True)

# Define missing values list
missing_values = [
    'na', 'n/a', 'N/A', 'NAN', 'NA', 'Null', 'null', 'NULL', 
    'Nan', 'nan', 'Unknown', 'unknown', 'UNKNOWN', '-', '--', 
    '---', '----', '', ' ', '  ', '   ', '    ', '?', '??', 
    '???', '????', 'Missing', 'missing', 'MISSING'
]

# Custom transformer for mode imputation
class ModeImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.modes_ = X.mode().iloc[0]  
        return self

    def transform(self, X):
        return X.fillna(self.modes_)

def update_progress(filename, progress, message=""):
    """Update the progress status of a file processing task"""
    status_path = status_folder / f"{filename}_status.json"
    status = {
        "filename": filename,
        "progress": progress,  # 0-100 percentage
        "status": "in_progress" if progress < 100 else "completed",
        "message": message,
        "last_updated": time.time()
    }
    with open(status_path, 'w') as f:
        json.dump(status, f)
    return status

def preprocess_file(file_path, output_path):
    """
    Preprocess a CSV file and save the cleaned version with progress tracking.
    """
    filename = file_path.name
    try:
        # Initialize progress
        update_progress(filename, 0, "Starting preprocessing")
        
        # Read the CSV file
        update_progress(filename, 5, "Reading CSV file")
        df = pl.read_csv(file_path, null_values=missing_values, ignore_errors=True)
        df = df.to_pandas()
        
        # Store original columns
        original_columns = list(df.columns)
        
        # Clean data
        update_progress(filename, 15, "Cleaning data and removing nulls")
        # Remove white space
        df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
        
        # Track missing values statistics
        missing_value_stats = {}
        for column in df.columns:
            missing_count = df[column].isna().sum()
            missing_percentage = round((missing_count / len(df)) * 100, 2)
            if missing_count > 0:
                missing_value_stats[column] = {
                    "missing_count": int(missing_count),
                    "missing_percentage": missing_percentage,
                    "imputation_method": "None"
                }
        
        # Drop rows with all null values
        df.dropna(how='all', inplace=True)
        
        # Drop columns with more than 95% Null values
        to_drop = [column for column in df.columns if (('duplicated') in column or column=='') 
                   and df[column].isna().sum() > (df.shape[0] * 0.95)]
        df.drop(to_drop, axis=1, inplace=True)
        
        # Create a copy for imputation
        update_progress(filename, 25, "Preparing for imputation")
        df2 = df.copy()
        
        # Identify column types
        nominal_columns = [column for column in df2.columns if df2[column].dtype in ['object', 'category', 'string', 'bool']]
        non_nominal_columns = [column for column in df2.columns if column not in nominal_columns and df[column].dtype != 'datetime64[ns]']
        
        # Track which columns will be imputed
        columns_with_imputation = []
        
        # Create and apply preprocessing pipeline
        update_progress(filename, 35, "Creating preprocessing pipeline")
        nominal_imputer = ModeImputer()
        non_nominal_imputer = KNNImputer()
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('nominal_imputing', nominal_imputer, nominal_columns),
                ('non_nominal_imputing', non_nominal_imputer, non_nominal_columns)
            ],
            remainder='passthrough'
        )
        
        # Apply preprocessing
        update_progress(filename, 45, "Imputing missing values")
        if nominal_columns and non_nominal_columns:
            df2 = preprocessor.fit_transform(df2)
            
            # Update imputation method in missing value stats
            for column in nominal_columns:
                if column in missing_value_stats:
                    missing_value_stats[column]["imputation_method"] = "Mode Imputation"
                    columns_with_imputation.append(column)
                    
            for column in non_nominal_columns:
                if column in missing_value_stats:
                    missing_value_stats[column]["imputation_method"] = "KNN Imputation"
                    columns_with_imputation.append(column)
            
            df2_imputed = pd.DataFrame(df2, columns=nominal_columns + non_nominal_columns)
            
            # Save the preprocessing pipeline
            update_progress(filename, 60, "Saving preprocessing pipeline")
            pipeline_path = processed_folder / f"{filename}_cleaning_pipeline.joblib"
            joblib.dump(preprocessor, pipeline_path)
        else:
            df2_imputed = df2
        
        # Process datetime columns
        update_progress(filename, 70, "Processing datetime columns")
        date_patterns = [
            r"\b\d{4}-\d{2}-\d{2}\b",
            r"\b\d{2}/\d{2}/\d{4}\b",
            r"\b\d{2}-\d{2}-\d{4}\b",
            r"\b\d{2}\.\d{2}\.\d{4}\b",
            r"\b\d{2}\.\d{2}\.\d{2}\b",
            r"\b\d{1,2}-[A-Za-z]{3}-\d{4}\b",
            r"\b[A-Za-z]+\s\d{1,2},\s\d{4}\b",
            r"\b\d{1,2}/\d{4}\b",
            r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?\.+\b",
        ]
        
        def detect_dates(series, patterns):
            if len(series) > 0 and isinstance(series.iloc[0], str):
                for pattern in patterns:
                    if re.search(pattern, series.iloc[0]):
                        return True
            return False
        
        # Check for date columns
        date_like = []
        for column in df2_imputed.columns:
            if column not in non_nominal_columns:
                if df2_imputed[column].dtype == 'object' and len(df2_imputed[column]) > 0:
                    if detect_dates(df2_imputed[column], date_patterns):
                        date_like.append(column)
        
        date_containing = [column for column in date_like if "date" in column.lower()]
        
        # Convert date columns
        update_progress(filename, 80, "Converting date columns")
        for column in date_containing:
            df2_imputed[column] = pd.to_datetime(df2_imputed[column], errors='ignore')
        
        # Drop columns with only one unique value
        update_progress(filename, 90, "Finalizing data")
        columns_before_final_drop = list(df2_imputed.columns)
        df2_imputed = df2_imputed.drop(columns=[col for col in df2_imputed.columns if df2_imputed[col].nunique() == 1])
        dropped_by_unique = list(set(columns_before_final_drop) - set(df2_imputed.columns))
        
        # Save processed file
        df2_imputed.to_csv(output_path, index=False)
        update_progress(filename, 100, "Processing completed successfully")
        
        # Calculate which columns were dropped
        final_columns = list(df2_imputed.columns)
        all_dropped_columns = list(set(original_columns) - set(final_columns))
        
        result = {
            "success": True,
            "original_shape": df.shape,
            "processed_shape": df2_imputed.shape,
            "columns_dropped": all_dropped_columns,
            "date_columns_detected": date_containing,
            "columns_cleaned": columns_with_imputation,
            "missing_value_stats": missing_value_stats,
            "dropped_by_unique_value": dropped_by_unique
        }
        
        # Save result details
        result_path = processed_folder / f"{filename}_results.json"
        with open(result_path, 'w') as f:
            # Convert tuple shapes to lists for JSON serialization
            result["original_shape"] = list(result["original_shape"])
            result["processed_shape"] = list(result["processed_shape"])
            json.dump(result, f)
            
        return result
    
    except Exception as e:
        error_message = str(e)
        update_progress(filename, -1, f"Error: {error_message}")
        
        return {
            "success": False,
            "error": error_message
        }

@app.post("/upload/")
async def upload_files(background_tasks: BackgroundTasks, files: list[UploadFile] = File(...)):
    saved_files = []
    processing_info = []
    
    for file in files:
        # Check file type
        is_csv = file.filename.lower().endswith('.csv')
        is_xlsx = file.filename.lower().endswith('.xlsx')
        
        if not (is_csv or is_xlsx):
            raise HTTPException(
                status_code=400, 
                detail=f"File type not supported for {file.filename}. Only CSV or XLSX allowed."
            )
        
        # Generate a unique filename to prevent collisions
        unique_id = str(uuid.uuid4())[:8]
        safe_filename = f"{unique_id}_{file.filename}"
        
        # Save the file to the local folder
        file_path = upload_folder / safe_filename
        with file_path.open("wb") as f:
            content = await file.read()
            f.write(content)
        
        saved_files.append({
            "original_filename": file.filename,
            "saved_filename": safe_filename
        })
        
        # Process the file in the background
        if is_csv:
            output_filename = f"processed_{safe_filename}"
            output_path = processed_folder / output_filename
            
            # Initialize progress tracking
            init_status = update_progress(safe_filename, 0, "Queued for processing")
            processing_info.append({
                "filename": safe_filename,
                "original_filename": file.filename,
                "processed_filename": output_filename,
                "status": init_status
            })
            
            background_tasks.add_task(preprocess_file, file_path, output_path)
        
    return JSONResponse(
        content={
            "message": f"{len(saved_files)} file(s) uploaded successfully.",
            "files": saved_files,
            "processing": "File preprocessing started in the background. Track progress using the /processing-status/{filename} endpoint.",
            "processing_info": processing_info
        }
    )

@app.get("/processed-files/")
async def get_processed_files():
    """Get a list of all processed files"""
    files = [f.name for f in processed_folder.glob("*") if f.is_file() and not (f.name.endswith('.joblib') or f.name.endswith('.json'))]
    return {"processed_files": files}

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Get a processed file by filename"""
    file_path = processed_folder / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return {"file_url": f"/processed_files/{filename}"}

@app.get("/processing-status/{filename}")
async def get_processing_status(filename: str):
    """Get the processing status and progress of a file"""
    status_path = status_folder / f"{filename}_status.json"
    result_path = processed_folder / f"{filename}_results.json"
    
    if status_path.exists():
        with open(status_path, 'r') as f:
            status = json.load(f)
            
        # If processing is complete, include results information
        if status["progress"] == 100 and result_path.exists():
            with open(result_path, 'r') as f:
                results = json.load(f)
                status["results"] = results
                
        return status
    else:
        return {"status": "not_found", "message": f"No processing status found for {filename}"}

@app.post("/custom-preprocess/")
async def custom_preprocess(
    file_id: str = Form(...),
    operations: str = Form(...),
    user_id: str = Form(...)
):
    """Apply custom preprocessing operations to a file"""
    # This is maintained for compatibility, but new code should use the /custom-preprocessing/ routes
    try:
        operations_list = json.loads(operations)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid operations format: {str(e)}")
    
    # Mock response for demonstration
    return {
        "success": True,
        "file_id": file_id,
        "operations_applied": operations_list,
        "message": "Custom preprocessing completed successfully",
        "result_file": f"custom_processed_{file_id}.csv"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)