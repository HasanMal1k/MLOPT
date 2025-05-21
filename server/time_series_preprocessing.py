from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pathlib import Path
import pandas as pd
import numpy as np
import polars as pl
import json
import uuid
import os
import re
from typing import List, Optional, Dict, Any, Union
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("time_series_preprocessing")

# Create router for time series preprocessing
router = APIRouter(prefix="/time-series")

# Create folders to store results if they don't exist
result_folder = Path("time_series_results")
result_folder.mkdir(exist_ok=True)

# Comprehensive list of missing value indicators
MISSING_VALUES = [
    'na', 'n/a', 'N/A', 'NAN', 'nan', 'NA', 'Null', 'null', 'NULL', 
    'Nan', 'Unknown', 'unknown', 'UNKNOWN', '', '-', '--', '---', '----', 
    ' ', '  ', '   ', '    ', '?', '??', '???', '????', 'Missing', 
    'missing', 'MISSING', 'None'
]

# Date patterns for detection
DATE_PATTERNS = [
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\s*(?:\d{1,2}:\d{2}(?::\\d{2})?(?:\s*?[APap][Mm])?)?\\b",
    r"\b\d{1,2}-\d{1,2}-\d{2,4}\s*(?:[ T]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?)?\\b",
    r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\s*(?:[ T]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?)?\\b",
    r"\b\d{1,2}\.\d{1,2}\.\d{2}\s*(?:[ T]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?)?\\b",
    r"\b\d{1,2}-[A-Za-z]{3}-\d{4}\s*(?:[ T]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?)?\\b",
    r"\b[A-Za-z]+\s\d{1,2},\s\d{4}\s*\d{1,2}:\d{2}(?::\d{2})?(?: ?[APap][Mm])?\\b",
    r"\b\d{1,2}/\d{4} (?:[ T]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?)?\\b",
    r"\b\d{4}-\d{1,2}-\d{1,2}\s*(?:[ T]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?)?\\b",
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\s*\d{1,2}:\d{2}(?::\d{2})?(?: ?[APap][Mm])?\\b",
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\s*(?:\d{1,2}:\d{2}(?::\d{2})?(?: ?[APap][Mm])?)?\\b",
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\s*\d{1,2}:\d{2}(?::\d{2})?\s?[APap][Mm]\\b",
    r"\b\d{2}/\d{2}/\d{4}\s*\d{1,2}:\d{2}:\d{2}\s?([APap][Mm])?\\b"
]

@router.post("/analyze-time-series/")
async def analyze_time_series(file: UploadFile = File(...)):
    """
    Analyze a file to determine if it's suitable for time series processing and
    identify potential date/time columns
    """
    try:
        # Save the file temporarily
        file_content = await file.read()
        temp_file_path = f"temp_{uuid.uuid4()}.csv"
        
        with open(temp_file_path, "wb") as f:
            f.write(file_content)
        
        # Try to read with polars for better performance
        try:
            df = pl.read_csv(temp_file_path, null_values=MISSING_VALUES, ignore_errors=True)
            df = df.to_pandas()
        except Exception:
            # Fall back to pandas if polars fails
            df = pd.read_csv(temp_file_path)
        
        # Clean up temporary file
        os.remove(temp_file_path)
        
        # Replace empty strings and spaces with NaN
        df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
        
        # Detect potential date columns
        date_columns = detect_date_columns(df)
        
        # Get data types of all columns
        column_types = {}
        for column in df.columns:
            column_types[column] = str(df[column].dtype)
        
        # Get basic statistics
        column_stats = {}
        for column in df.columns:
            # Skip date columns for now
            if column in date_columns:
                continue
                
            # For numerical columns, get statistics
            if pd.api.types.is_numeric_dtype(df[column]):
                column_stats[column] = {
                    "min": float(df[column].min()) if not pd.isna(df[column].min()) else None,
                    "max": float(df[column].max()) if not pd.isna(df[column].max()) else None,
                    "mean": float(df[column].mean()) if not pd.isna(df[column].mean()) else None,
                    "missing": int(df[column].isna().sum()),
                    "missing_pct": float(df[column].isna().mean() * 100)
                }
            else:
                # For non-numeric columns, get unique values count
                column_stats[column] = {
                    "unique_values": int(df[column].nunique()),
                    "missing": int(df[column].isna().sum()),
                    "missing_pct": float(df[column].isna().mean() * 100)
                }
        
        # Determine if this is likely a time series dataset
        # Criteria: Has at least one date column
        is_time_series = len(date_columns) > 0
        
        # Prepare recommendations
        recommendations = {
            "is_time_series": is_time_series,
            "date_columns": date_columns,
            "recommended_date_column": date_columns[0] if date_columns else None,
            "columns_with_missing_values": [col for col, stats in column_stats.items() 
                                           if stats["missing"] > 0]
        }
        
        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "row_count": len(df),
            "column_count": len(df.columns),
            "column_types": column_types,
            "column_stats": column_stats,
            "date_columns": date_columns,
            "recommendations": recommendations
        })
        
    except Exception as e:
        logger.error(f"Error analyzing time series: {str(e)}")
        return JSONResponse(
            status_code=500,
            detail=f"Error analyzing file: {str(e)}"
        )

def detect_date_columns(df):
    """Detect columns that likely contain date information"""
    date_columns = []
    
    # First check for columns already in datetime format
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    date_columns.extend(datetime_cols)
    
    # Then check for columns with 'date' in the name
    date_containing = [column for column in df.columns 
                       if "date" in column.lower() and column not in date_columns]
    
    # For each column that has 'date' in the name, check if it matches date patterns
    for column in date_containing:
        if column in df.columns and df[column].dtype in ['object', 'string']:
            try:
                # Try to convert a sample to datetime
                sample = df[column].dropna().head(5)
                if len(sample) > 0 and any(detect_dates(str(val), DATE_PATTERNS) for val in sample):
                    date_columns.append(column)
            except:
                pass
    
    # For remaining string columns, check for date patterns
    for column in df.select_dtypes(include=['object', 'string']).columns:
        if column not in date_columns:
            try:
                sample = df[column].dropna().head(5)
                if len(sample) > 0 and any(detect_dates(str(val), DATE_PATTERNS) for val in sample):
                    date_columns.append(column)
            except:
                pass
    
    return date_columns

def detect_dates(value, patterns):
    """Check if a value matches any date patterns"""
    for pattern in patterns:
        if re.search(pattern, value):
            return True
    return False

def smart_datetime_parse(series):
    """
    Intelligently attempt to parse a series to datetime format
    trying both with and without dayfirst option
    """
    # Try with dayfirst=False (MM/DD/YYYY)
    dt1 = pd.to_datetime(series, errors='coerce', dayfirst=False)
    errors1 = dt1.isna().sum()
    
    # Try with dayfirst=True (DD/MM/YYYY)
    dt2 = pd.to_datetime(series, errors='coerce', dayfirst=True)
    errors2 = dt2.isna().sum()
    
    # Return the version with fewer errors
    if errors1 <= errors2:
        return dt1
    else:
        return dt2

@router.post("/process-time-series/")
async def process_time_series(
    file: UploadFile = File(...),
    date_column: str = Form(...),
    frequency: str = Form(...),
    drop_columns: Optional[str] = Form(None)
):
    """
    Process a time series file:
    1. Identify and convert date column
    2. Set the date column as index
    3. Impute missing values
    4. Set frequency
    5. Convert all columns to float
    """
    try:
        # Save the file temporarily
        file_content = await file.read()
        temp_file_path = f"temp_{uuid.uuid4()}.csv"
        
        with open(temp_file_path, "wb") as f:
            f.write(file_content)
        
        # Parse drop_columns if provided
        columns_to_drop = json.loads(drop_columns) if drop_columns else []
        
        # Read the file
        try:
            df = pl.read_csv(temp_file_path, null_values=MISSING_VALUES, ignore_errors=True)
            df = df.to_pandas()
        except Exception:
            # Fall back to pandas if polars fails
            df = pd.read_csv(temp_file_path)
        
        # Track processing steps for reporting
        processing_steps = []
        processing_steps.append(f"Loaded file with {len(df)} rows and {len(df.columns)} columns")
        
        # 1. Initial cleaning
        # Replace empty strings with NaN
        df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
        # Drop rows with all NaN values
        initial_rows = len(df)
        df.dropna(how='all', inplace=True)
        dropped_rows = initial_rows - len(df)
        processing_steps.append(f"Dropped {dropped_rows} completely empty rows")
        
        # 2. Handle date column
        if date_column not in df.columns:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": f"Date column '{date_column}' not found in dataset"}
            )
        
        # Convert date column to datetime using smart parse
        df[date_column] = smart_datetime_parse(df[date_column])
        processing_steps.append(f"Converted '{date_column}' to datetime format")
        
        # Drop columns if specified
        if columns_to_drop:
            df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
            processing_steps.append(f"Dropped {len(columns_to_drop)} columns: {', '.join(columns_to_drop)}")
        
        # 3. Set date column as index
        df = df.set_index(date_column)
        processing_steps.append(f"Set '{date_column}' as index")
        
        # 4. Handle missing values using smart imputation
        missing_before = df.isna().sum().sum()
        for col in df.columns:
            if df[col].isna().sum() > 0:
                df[col] = smart_fillna_timeseries(df[col])
        
        missing_after = df.isna().sum().sum()
        processing_steps.append(f"Imputed {missing_before - missing_after} missing values")
        
        # 5. Set frequency
        valid_frequencies = ['D', 'B', 'W', 'M', 'Q', 'A', 'H', 'T', 'S']
        if frequency not in valid_frequencies:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": f"Invalid frequency '{frequency}'. Valid options are: {', '.join(valid_frequencies)}"}
            )
        
        # Convert to specified frequency
        df_freq = df.asfreq(frequency)
        processing_steps.append(f"Set frequency to '{frequency}'")
        
        # 6. Drop rows with all missing values (created by frequency resampling)
        freq_rows_before = len(df_freq)
        df_freq.dropna(how='all', inplace=True)
        freq_rows_dropped = freq_rows_before - len(df_freq)
        processing_steps.append(f"Dropped {freq_rows_dropped} empty rows after frequency conversion")
        
        # 7. Convert all columns to float
        # First remove commas that might prevent conversion
        df_freq = df_freq.replace(r',', '', regex=True)
        
        # Track columns that couldn't be converted to float
        failed_float_conversions = []
        
        for col in df_freq.columns:
            try:
                df_freq[col] = df_freq[col].astype(float)
            except:
                failed_float_conversions.append(col)
        
        if failed_float_conversions:
            processing_steps.append(f"Warning: Could not convert these columns to float: {', '.join(failed_float_conversions)}")
        else:
            processing_steps.append("Successfully converted all columns to float type")
        
        # 8. Final check for any remaining missing values
        remaining_missing = df_freq.isna().sum().sum()
        if remaining_missing > 0:
            columns_with_missing = [col for col in df_freq.columns if df_freq[col].isna().sum() > 0]
            processing_steps.append(f"Warning: {remaining_missing} missing values remain in columns: {', '.join(columns_with_missing)}")
        
        # 9. Save the processed file
        unique_id = str(uuid.uuid4())[:8]
        output_filename = f"ts_processed_{unique_id}_{file.filename}"
        output_path = result_folder / output_filename
        
        # Reset index to include date column in the saved file
        df_freq = df_freq.reset_index()
        df_freq.to_csv(output_path, index=False)
        
        # Generate a report of statistics
        stats = {
            "original_rows": initial_rows,
            "processed_rows": len(df_freq),
            "original_columns": len(df.columns) + 1,  # +1 for the date column that became index
            "processed_columns": len(df_freq.columns),
            "frequency": frequency,
            "date_column": date_column,
            "missing_values_handled": missing_before - missing_after,
            "columns_with_issues": failed_float_conversions
        }
        
        # Clean up temporary file
        os.remove(temp_file_path)
        
        return JSONResponse(content={
            "success": True,
            "message": "Time series processing complete",
            "filename": file.filename,
            "processed_filename": output_filename,
            "processing_steps": processing_steps,
            "statistics": stats
        })
        
    except Exception as e:
        logger.error(f"Error processing time series: {str(e)}")
        # Clean up if temporary file exists
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Error processing time series: {str(e)}"}
        )

def smart_fillna_timeseries(series: pd.Series, rolling_window=3, interp_method='linear', fill_limit=None):
    """
    Fills NaNs in a time series with a flexible strategy:
    1. Rolling mean for short gaps
    2. Interpolation for medium gaps
    3. Forward/backward fill for remaining gaps
    """
    # Start with a copy to avoid modifying the original
    filled = series.copy()
    
    # Step 1: Rolling mean
    rolling = series.rolling(window=rolling_window, min_periods=1).mean()
    filled = filled.fillna(rolling)
    
    # Step 2: Interpolation
    filled = filled.interpolate(method=interp_method, limit_direction='both')
    
    # Step 3: Final fallback fill (forward/backward fill)
    filled = filled.fillna(method='ffill', limit=fill_limit)
    filled = filled.fillna(method='bfill', limit=fill_limit)
    
    return filled

@router.post("/get-frequencies/")
async def get_frequencies():
    """Return available time series frequencies with descriptions"""
    frequencies = [
        {"code": "D", "description": "Calendar day frequency"},
        {"code": "B", "description": "Business day frequency"},
        {"code": "W", "description": "Weekly frequency"},
        {"code": "M", "description": "Month end frequency"},
        {"code": "Q", "description": "Quarter end frequency"},
        {"code": "A", "description": "Year end frequency"},
        {"code": "H", "description": "Hourly frequency"},
        {"code": "T", "description": "Minute frequency"},
        {"code": "S", "description": "Second frequency"}
    ]
    
    return JSONResponse(content={
        "success": True,
        "frequencies": frequencies
    })

@router.post("/custom-impute-time-series/")
async def custom_impute_time_series(
    file: UploadFile = File(...),
    date_column: str = Form(...),
    imputation_method: str = Form("auto"),
    imputation_config: Optional[str] = Form(None)
):
    """
    Apply custom imputation methods to time series data
    
    Methods:
    - auto: Use smart imputation strategy
    - interpolate: Linear interpolation
    - mean: Use mean value of the column
    - median: Use median value of the column
    - ffill: Forward fill
    - bfill: Backward fill
    - rolling: Rolling window average
    - custom: Custom configuration per column
    """
    try:
        # Save the file temporarily
        file_content = await file.read()
        temp_file_path = f"temp_{uuid.uuid4()}.csv"
        
        with open(temp_file_path, "wb") as f:
            f.write(file_content)
        
        # Read the file
        try:
            df = pl.read_csv(temp_file_path, null_values=MISSING_VALUES, ignore_errors=True)
            df = df.to_pandas()
        except Exception:
            # Fall back to pandas if polars fails
            df = pd.read_csv(temp_file_path)
        
        # Clean up temporary file
        os.remove(temp_file_path)
        
        # Convert date column and set as index
        if date_column not in df.columns:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": f"Date column '{date_column}' not found in dataset"}
            )
        
        # Convert date column to datetime
        df[date_column] = smart_datetime_parse(df[date_column])
        
        # Set date column as index
        df = df.set_index(date_column)
        
        # Parse imputation config if provided
        column_configs = {}
        if imputation_config:
            try:
                column_configs = json.loads(imputation_config)
            except:
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "error": "Invalid imputation configuration format"}
                )
        
        # Track imputation results
        imputation_results = {}
        
        # Apply imputation method
        for column in df.columns:
            # Skip columns with no missing values
            if df[column].isna().sum() == 0:
                continue
                
            missing_before = df[column].isna().sum()
            
            # Determine method for this column
            method = imputation_method
            if column in column_configs:
                method = column_configs[column].get('method', imputation_method)
            
            # Apply the appropriate method
            if method == "auto":
                df[column] = smart_fillna_timeseries(df[column])
            
            elif method == "interpolate":
                df[column] = df[column].interpolate(method='linear', limit_direction='both')
            
            elif method == "mean":
                mean_val = df[column].mean()
                df[column] = df[column].fillna(mean_val)
            
            elif method == "median":
                median_val = df[column].median()
                df[column] = df[column].fillna(median_val)
            
            elif method == "ffill":
                df[column] = df[column].fillna(method='ffill')
            
            elif method == "bfill":
                df[column] = df[column].fillna(method='bfill')
            
            elif method == "rolling":
                window_size = 3  # Default
                if column in column_configs:
                    window_size = column_configs[column].get('window_size', 3)
                
                rolling = df[column].rolling(window=window_size, min_periods=1).mean()
                df[column] = df[column].fillna(rolling)
            
            # Track results
            missing_after = df[column].isna().sum()
            imputation_results[column] = {
                "method": method,
                "missing_before": int(missing_before),
                "missing_after": int(missing_after),
                "filled_values": int(missing_before - missing_after)
            }
        
        # Save the processed file
        unique_id = str(uuid.uuid4())[:8]
        output_filename = f"ts_imputed_{unique_id}_{file.filename}"
        output_path = result_folder / output_filename
        
        # Reset index to include date column in the saved file
        df = df.reset_index()
        df.to_csv(output_path, index=False)
        
        return JSONResponse(content={
            "success": True,
            "message": "Time series imputation complete",
            "filename": file.filename,
            "processed_filename": output_filename,
            "imputation_results": imputation_results,
            "total_values_imputed": sum(result["filled_values"] for result in imputation_results.values())
        })
        
    except Exception as e:
        logger.error(f"Error in custom imputation: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Error in custom imputation: {str(e)}"}
        )