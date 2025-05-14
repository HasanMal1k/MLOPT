from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse
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
import warnings
import tempfile
from pandas.errors import PerformanceWarning
from ydata_profiling import ProfileReport


# Import custom preprocessing module
from custom_preprocessing import router as custom_preprocessing_router
from transformations import router as transformations_router



app = FastAPI()

# Register custom preprocessing router
app.include_router(custom_preprocessing_router)
app.include_router(transformations_router)


origins = [
    "http://127.0.0.1:3000",
    "http://localhost:3001",  # Next.js app URL
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
    Fixed version to eliminate warnings and improve performance.
    """
    
    # Suppress specific warnings
    warnings.filterwarnings('ignore', category=PerformanceWarning)
    
    result = {}
    filename = file_path.name
    
    try:
        # Initialize progress
        update_progress(filename, 0, "Starting preprocessing")
        
        # Read the CSV file
        update_progress(filename, 5, "Reading CSV file")
        try:
            df = pl.read_csv(file_path, null_values=missing_values, ignore_errors=True)
            df = df.to_pandas()
        except Exception as e:
            # Fallback to pandas directly
            df = pd.read_csv(file_path, na_values=missing_values)
        
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
            r"\b\d{1,2}/\d{1,2}/\d{2,4}\s*(?:\d{1,2}:\d{2}(?::\d{2})?(?:\s*?[APap][Mm])?)?\b",
            r"\b\d{1,2}-\d{1,2}-\d{2,4}\s*(?:[ T]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?)?\b",
            r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\s*(?:[ T]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?)?\b",
            r"\b\d{1,2}\.\d{1,2}\.\d{2}\s*(?:[ T]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?)?\b",
            r"\b\d{1,2}-[A-Za-z]{3}-\d{4}\s*(?:[ T]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?)?\b",
            r"\b[A-Za-z]+\s\d{1,2},\s\d{4}\s*\d{1,2}:\d{2}(?::\d{2})?(?: ?[APap][Mm])?\b",
            r"\b\d{1,2}/\d{4} (?:[ T]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?)?\b",
            r"\b\d{4}-\d{1,2}-\d{1,2}\s*(?:[ T]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?)?\b",
            r"\b\d{1,2}/\d{1,2}/\d{2,4}\s*\d{1,2}:\d{2}(?::\d{2})?(?: ?[APap][Mm])?\b",
            r"\b\d{1,2}/\d{1,2}/\d{2,4}\s*(?:\d{1,2}:\d{2}(?::\d{2})?(?: ?[APap][Mm])?)?\b",
            r"\b\d{1,2}/\d{1,2}/\d{2,4}\s*\d{1,2}:\d{2}(?::\d{2})?\s?[APap][Mm]\b",
            r"\b\d{2}/\d{2}/\d{4}\s*\d{1,2}:\d{2}:\d{2}\s?([APap][Mm])?\b"
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
        
        # Convert date columns safely without warnings
        update_progress(filename, 80, "Converting date columns")
        for column in date_containing:
            try:
                df2_imputed[column] = pd.to_datetime(df2_imputed[column], errors='coerce')
            except Exception as e:
                print(f"Could not convert {column} to datetime: {e}")
        
        # Use improved engineer_features function
        update_progress(filename, 85, "Generating engineered features")
        try:
            from transformations import engineer_features
            df2_imputed, transformation_results = engineer_features(df2_imputed)
            
            # Track which features were engineered with improved extraction
            engineered_features = []
            for feature_type, items in transformation_results.items():
                if isinstance(items, list) and len(items) > 0:
                    for item in items:
                        if feature_type == "datetime_features" and 'derived_features' in item:
                            engineered_features.extend(item["derived_features"])
                        elif feature_type == "categorical_encodings" and 'derived_features' in item:
                            engineered_features.extend(item["derived_features"])
                        elif feature_type == "numeric_transformations" and 'derived_features' in item:
                            engineered_features.extend(item["derived_features"])
                        elif feature_type == "binned_features" and 'derived_feature' in item:
                            engineered_features.append(item["derived_feature"])
                        
            result["engineered_features"] = engineered_features
            result["transformation_details"] = transformation_results
        except Exception as e:
            print(f"Feature engineering error: {e}")
            result["engineered_features"] = []
            result["transformation_details"] = {}
        
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


# Additional helper function for safer datetime conversion
def safe_datetime_conversion(column, method='coerce'):
    """
    Safely convert a column to datetime with proper error handling
    """
    try:
        # Try standard conversion first
        converted = pd.to_datetime(column, errors=method)
        
        # If too few values converted successfully, return original
        if method == 'coerce' and converted.notna().sum() < 0.1 * len(column):
            return column
            
        return converted
    except Exception as e:
        print(f"Datetime conversion failed: {e}")
        return column


# Alternative improved preprocessing function with minimal warnings
def preprocess_file_optimized(file_path, output_path):
    """
    Optimized version of preprocess_file with minimal warnings
    """
    import warnings
    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
    warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
    
    filename = file_path.name
    
    try:
        # Read CSV with better error handling
        try:
            df = pl.read_csv(file_path, null_values=missing_values, ignore_errors=True)
            df = df.to_pandas()
        except:
            df = pd.read_csv(file_path, na_values=missing_values)
            
        # Track original state
        original_columns = df.columns.tolist()
        original_shape = df.shape
        
        # Efficient cleaning pipeline
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Clean data efficiently
            df = df.replace(r'^\s*$', np.nan, regex=True)
            df = df.dropna(how='all')
            
            # Drop problematic columns
            to_drop = [col for col in df.columns if 
                      'duplicated' in col or 
                      col == '' or 
                      df[col].isna().sum() > (df.shape[0] * 0.95)]
            df.drop(to_drop, axis=1, inplace=True)
        
        # Imputation with minimal warnings
        df_imputed = perform_imputation(df)
        
        # DateTime conversion
        df_imputed = convert_datetime_columns(df_imputed)
        
        # Feature engineering
        df_final, transform_results = engineer_features(df_imputed)
        
        # Final cleanup
        df_final = df_final.drop(columns=[col for col in df_final.columns 
                                         if df_final[col].nunique() == 1])
        
        # Save results
        df_final.to_csv(output_path, index=False)
        
        return generate_results(df, df_final, original_columns, transform_results)
        
    except Exception as e:
        return {"success": False, "error": str(e)}


def perform_imputation(df):
    """Perform imputation with reduced warnings"""
    nominal_cols = df.select_dtypes(include=['object', 'category', 'string', 'bool']).columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        if nominal_cols.any():
            df[nominal_cols] = df[nominal_cols].fillna(df[nominal_cols].mode().iloc[0])
        
        if numeric_cols.any():
            imputer = KNNImputer()
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    return df


def convert_datetime_columns(df):
    """Convert datetime columns efficiently"""
    # Batch convert potential date columns
    potential_date_cols = [col for col in df.columns 
                          if df[col].dtype == 'object' and 
                          any(keyword in col.lower() for keyword in ['date', 'time', 'timestamp'])]
    
    for col in potential_date_cols:
        try:
            converted = pd.to_datetime(df[col], errors='coerce')
            if converted.notna().sum() > 0.5 * len(df):
                df[col] = converted
        except:
            pass
    
    return df


def generate_results(df_orig, df_final, original_columns, transform_results):
    """Generate result dictionary with proper formatting"""
    final_columns = df_final.columns.tolist()
    dropped_columns = list(set(original_columns) - set(final_columns))
    
    return {
        "success": True,
        "original_shape": list(df_orig.shape),
        "processed_shape": list(df_final.shape),
        "columns_dropped": dropped_columns,
        "engineered_features": extract_engineered_features(transform_results),
        "transformation_details": transform_results
    }


def extract_engineered_features(transform_results):
    """Extract engineered features from transformation results"""
    engineered_features = []
    
    for feature_type, items in transform_results.items():
        if isinstance(items, list):
            for item in items:
                if 'derived_features' in item:
                    engineered_features.extend(item['derived_features'])
                elif 'derived_feature' in item:
                    engineered_features.append(item['derived_feature'])
    
    return engineered_features

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

@app.post("/generate_report/", response_class=HTMLResponse)
async def generate_report(file: UploadFile = File(...)):
    # Only accept CSV/XLSX
    if not file.filename.endswith(('.csv', '.xlsx')):
        raise HTTPException(status_code=400, detail="Only CSV or XLSX files are accepted")

    try:
        # Read the file content
        contents = await file.read()
        
        # Log file details for debugging
        print(f"Received file: {file.filename}, size: {len(contents)} bytes")
        
        # Create a temporary file
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            temp_path = tmp.name
        
        # Read the file with pandas - with better error handling
        try:
            if suffix.lower() == '.csv':
                # Try different options to handle potential CSV issues
                try:
                    df = pd.read_csv(temp_path)
                except pd.errors.EmptyDataError:
                    # If it's empty, create a sample dataframe to avoid crashing
                    df = pd.DataFrame({'Sample': ['No data found in file']})
                except:
                    # Try again with different parsing options
                    df = pd.read_csv(temp_path, sep=None, engine='python')
            else:
                # Excel file
                df = pd.read_excel(temp_path)
            
            # Limit the rows to prevent very large reports
            if len(df) > 10000:
                df = df.sample(10000, random_state=42)
                
            # Generate the profile report
            profile = ProfileReport(df, explorative=True, minimal=True)
            html_report = profile.to_html()
        except Exception as e:
            # If pandas couldn't read the file properly, return a simple error report
            error_info = str(e)
            html_report = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Error Processing File</title>
                <style>
                    body {{ font-family: Arial, sans-serif; padding: 20px; }}
                    .error {{ color: red; background-color: #ffeeee; padding: 10px; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <h1>File Processing Error</h1>
                <p>There was an error processing the file: {file.filename}</p>
                <div class="error">
                    <h3>Error Details:</h3>
                    <pre>{error_info}</pre>
                </div>
            </body>
            </html>
            """
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    # Return the HTML with proper headers
    return HTMLResponse(
        content=html_report,
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Content-Type": "text/html"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)