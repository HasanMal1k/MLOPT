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
import warnings
import math

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

# Improved date patterns for detection (fixed regex patterns)
DATE_PATTERNS = [
    r'\b\d{4}-\d{1,2}-\d{1,2}\b',           # YYYY-MM-DD (like 2017-01-01)
    r'\b\d{1,2}/\d{1,2}/\d{4}\b',           # MM/DD/YYYY or DD/MM/YYYY
    r'\b\d{1,2}-\d{1,2}-\d{4}\b',           # MM-DD-YYYY or DD-MM-YYYY  
    r'\b\d{4}/\d{1,2}/\d{1,2}\b',           # YYYY/MM/DD
    r'\b\d{1,2}\.\d{1,2}\.\d{4}\b',         # DD.MM.YYYY
    r'\b\d{4}\.\d{1,2}\.\d{1,2}\b',         # YYYY.MM.DD
    r'\b\d{1,2}-[A-Za-z]{3}-\d{4}\b',       # DD-MMM-YYYY
    r'\b[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}\b', # Month DD, YYYY
    r'\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}\b', # DD Month YYYY
    r'\b\d{4}-\d{1,2}-\d{1,2}T\d{2}:\d{2}:\d{2}\b', # ISO datetime
    r'\b\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}\b', # MM/DD/YYYY HH:MM
    r'\b\d{8}\b',                            # YYYYMMDD
]

# Column name patterns that suggest dates
DATE_COLUMN_PATTERNS = [
    r'(?i)^date$',           # Exact match for "date"
    r'(?i)^time$',           # Exact match for "time" 
    r'(?i)^datetime$',       # Exact match for "datetime"
    r'(?i).*date.*',         # Contains "date"
    r'(?i).*time.*',         # Contains "time"
    r'(?i).*year.*',         # Contains "year"
    r'(?i).*month.*',        # Contains "month"
    r'(?i).*day.*',          # Contains "day"
    r'(?i).*created.*',      # Contains "created"
    r'(?i).*modified.*',     # Contains "modified"
    r'(?i).*timestamp.*',    # Contains "timestamp"
    r'(?i).*period.*',       # Contains "period"
    r'(?i).*when.*',         # Contains "when"
    r'(?i).*posted.*',       # Contains "posted"
    r'(?i).*updated.*',      # Contains "updated"
    r'(?i).*occurred.*',     # Contains "occurred"
]

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

def convert_to_json_serializable(obj):
    """
    Convert numpy/pandas data types to JSON serializable Python types
    """
    return clean_for_json(obj)

def detect_dates(value: str) -> bool:
    """
    Check if a value matches any date patterns
    """
    if not value or pd.isna(value):
        return False
        
    value_str = str(value).strip()
    if not value_str:
        return False
    
    # Check against all patterns
    for pattern in DATE_PATTERNS:
        try:
            if re.search(pattern, value_str):
                return True
        except Exception:
            continue
    
    return False

def test_pandas_datetime_conversion(series: pd.Series, sample_size: int = 50) -> float:
    """
    Test how well a series converts to datetime and return success rate
    """
    sample = series.dropna().head(sample_size)
    if len(sample) == 0:
        return 0.0
    
    try:
        # Try multiple conversion methods
        methods = [
            lambda s: pd.to_datetime(s, errors='coerce'),
            lambda s: pd.to_datetime(s, errors='coerce', dayfirst=False),
            lambda s: pd.to_datetime(s, errors='coerce', dayfirst=True),
        ]
        
        best_success_rate = 0.0
        
        for method in methods:
            try:
                converted = method(sample)
                success_rate = converted.notna().sum() / len(sample)
                best_success_rate = max(best_success_rate, success_rate)
            except:
                continue
        
        return best_success_rate
        
    except Exception as e:
        logger.warning(f"Error testing datetime conversion: {e}")
        return 0.0

def detect_date_columns(df: pd.DataFrame) -> List[str]:
    """
    Improved date column detection with multiple methods and better logging
    """
    date_columns = []
    
    logger.info(f"Starting date detection for DataFrame with shape {df.shape}")
    logger.info(f"Column names: {list(df.columns)}")
    logger.info(f"Column dtypes: {df.dtypes.to_dict()}")
    
    # Method 1: Check for columns that are already datetime type
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    date_columns.extend(datetime_cols)
    logger.info(f"Found {len(datetime_cols)} existing datetime columns: {datetime_cols}")
    
    # Method 2: Check column names for date-like patterns (prioritize exact matches)
    for col in df.columns:
        if col in date_columns:
            continue
            
        logger.info(f"Checking column '{col}' for date-like name patterns")
        
        # Check exact matches first (higher priority)
        exact_matches = [r'(?i)^date$', r'(?i)^time$', r'(?i)^datetime$']
        is_exact_match = any(re.search(pattern, col) for pattern in exact_matches)
        
        # Check partial matches
        is_partial_match = any(re.search(pattern, col) for pattern in DATE_COLUMN_PATTERNS)
        
        if is_exact_match or is_partial_match:
            logger.info(f"Column '{col}' matches date pattern (exact: {is_exact_match}, partial: {is_partial_match})")
            
            # For string/object columns, verify content
            if df[col].dtype in ['object', 'string']:
                # Get sample values
                sample = df[col].dropna().astype(str).head(20).tolist()
                logger.info(f"Sample values from '{col}': {sample[:5]}")
                
                if sample:
                    # Test pattern matching
                    pattern_matches = sum(1 for val in sample if detect_dates(str(val)))
                    pattern_match_rate = pattern_matches / len(sample)
                    
                    logger.info(f"Pattern match rate for '{col}': {pattern_matches}/{len(sample)} = {pattern_match_rate:.2%}")
                    
                    # Test pandas conversion
                    pandas_success_rate = test_pandas_datetime_conversion(df[col])
                    logger.info(f"Pandas conversion success rate for '{col}': {pandas_success_rate:.2%}")
                    
                    # Accept if either method shows reasonable success
                    if pattern_match_rate > 0.2 or pandas_success_rate > 0.2:
                        date_columns.append(col)
                        logger.info(f"✓ Added '{col}' based on name pattern and content verification")
                        continue
            
            # For exact name matches (like 'date'), add even if content verification is unclear
            elif is_exact_match:
                logger.info(f"✓ Added '{col}' based on exact name match ('{col}'), will attempt conversion later")
                date_columns.append(col)
                continue
    
    # Method 3: For remaining string columns, check content for date patterns
    remaining_string_cols = [col for col in df.select_dtypes(include=['object', 'string']).columns 
                            if col not in date_columns]
    
    logger.info(f"Checking {len(remaining_string_cols)} remaining string columns for date content")
    
    for col in remaining_string_cols:
        logger.info(f"Content analysis for column '{col}'")
        
        sample = df[col].dropna().astype(str).head(50).tolist()
        if not sample:
            logger.info(f"No non-null values in '{col}'")
            continue
        
        logger.info(f"Sample values from '{col}': {sample[:5]}")
        
        # Pattern matching test
        pattern_matches = sum(1 for val in sample if detect_dates(str(val)))
        pattern_match_rate = pattern_matches / len(sample)
        
        # Pandas conversion test
        pandas_success_rate = test_pandas_datetime_conversion(df[col])
        
        logger.info(f"'{col}' - Pattern matches: {pattern_matches}/{len(sample)} ({pattern_match_rate:.2%})")
        logger.info(f"'{col}' - Pandas conversion: {pandas_success_rate:.2%}")
        
        # Lower threshold for acceptance (20% instead of 30%)
        if pattern_match_rate > 0.2 or pandas_success_rate > 0.2:
            date_columns.append(col)
            logger.info(f"✓ Added '{col}' based on content analysis")
    
    # Method 4: Check for numeric columns that might be timestamps
    numeric_cols = [col for col in df.select_dtypes(include=['int64', 'float64']).columns
                   if col not in date_columns]
    
    logger.info(f"Checking {len(numeric_cols)} numeric columns for timestamp patterns")
    
    for col in numeric_cols:
        sample_values = df[col].dropna().head(20)
        if len(sample_values) == 0:
            continue
            
        min_val, max_val = sample_values.min(), sample_values.max()
        logger.info(f"Numeric column '{col}' range: {min_val} to {max_val}")
        
        # Check for Unix timestamps (seconds since epoch: 1970-2030)
        if 946684800 < min_val and max_val < 1893456000:
            try:
                converted = pd.to_datetime(sample_values, unit='s', errors='coerce')
                success_rate = converted.notna().sum() / len(sample_values)
                logger.info(f"Unix timestamp conversion success for '{col}': {success_rate:.2%}")
                
                if success_rate > 0.5:
                    date_columns.append(col)
                    logger.info(f"✓ Added '{col}' as Unix timestamp")
                    continue
            except Exception as e:
                logger.warning(f"Error testing Unix timestamp for '{col}': {e}")
        
        # Check for millisecond timestamps
        if 946684800000 < min_val and max_val < 1893456000000:
            try:
                converted = pd.to_datetime(sample_values, unit='ms', errors='coerce')
                success_rate = converted.notna().sum() / len(sample_values)
                logger.info(f"Millisecond timestamp conversion success for '{col}': {success_rate:.2%}")
                
                if success_rate > 0.5:
                    date_columns.append(col)
                    logger.info(f"✓ Added '{col}' as millisecond timestamp")
            except Exception as e:
                logger.warning(f"Error testing millisecond timestamp for '{col}': {e}")
    
    logger.info(f"Final result: {len(date_columns)} date columns detected: {date_columns}")
    
    # If no date columns found, provide detailed diagnostic info
    if not date_columns:
        logger.warning("No date columns detected. Diagnostic information:")
        for col in df.columns:
            sample = df[col].dropna().head(5)
            logger.warning(f"Column '{col}' (dtype: {df[col].dtype}): {list(sample)}")
    
    return date_columns

def smart_datetime_parse(series: pd.Series) -> pd.Series:
    """
    Intelligently attempt to parse a series to datetime format
    """
    logger.info(f"Parsing datetime series with {len(series)} values")
    
    # Handle numeric series (potential timestamps)
    if pd.api.types.is_numeric_dtype(series):
        # Check if values are in timestamp range
        non_null = series.dropna()
        if len(non_null) > 0:
            min_val, max_val = non_null.min(), non_null.max()
            
            # Unix timestamp (seconds)
            if 946684800 < min_val and max_val < 1893456000:
                try:
                    result = pd.to_datetime(series, unit='s', errors='coerce')
                    success_rate = result.notna().sum() / len(result)
                    logger.info(f"Unix timestamp conversion success rate: {success_rate:.2%}")
                    if success_rate > 0.5:
                        return result
                except:
                    pass
            
            # Millisecond timestamp
            elif 946684800000 < min_val and max_val < 1893456000000:
                try:
                    result = pd.to_datetime(series, unit='ms', errors='coerce')
                    success_rate = result.notna().sum() / len(result)
                    logger.info(f"Millisecond timestamp conversion success rate: {success_rate:.2%}")
                    if success_rate > 0.5:
                        return result
                except:
                    pass
    
    # For string data, try multiple parsing strategies
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Strategy 1: Try with dayfirst=False (US format)
        try:
            dt1 = pd.to_datetime(series, errors='coerce', dayfirst=False)
            errors1 = dt1.isna().sum()
        except:
            errors1 = len(series)
        
        # Strategy 2: Try with dayfirst=True (European format)
        try:
            dt2 = pd.to_datetime(series, errors='coerce', dayfirst=True)
            errors2 = dt2.isna().sum()
        except:
            errors2 = len(series)
        
        # Strategy 3: Try with infer_datetime_format (if available)
        try:
            dt3 = pd.to_datetime(series, errors='coerce', infer_datetime_format=True)
            errors3 = dt3.isna().sum()
        except:
            errors3 = len(series)
        
        # Choose the best result
        best_errors = min(errors1, errors2, errors3)
        
        if best_errors == errors1:
            result = dt1
        elif best_errors == errors2:
            result = dt2
        else:
            result = dt3
        
        success_rate = (len(result) - best_errors) / len(result)
        logger.info(f"Best datetime conversion success rate: {success_rate:.2%}")
        
        return result

@router.post("/analyze-time-series/")
async def analyze_time_series(file: UploadFile = File(...)):
    """
    Analyze a file to determine if it's suitable for time series processing and
    identify potential date/time columns
    """
    temp_file_path = None
    try:
        # Save the file temporarily
        file_content = await file.read()
        temp_file_path = f"temp_{uuid.uuid4()}.csv"
        
        with open(temp_file_path, "wb") as f:
            f.write(file_content)
        
        logger.info(f"Analyzing file: {file.filename}")
        
        # Try to read with polars for better performance
        try:
            df = pl.read_csv(temp_file_path, null_values=MISSING_VALUES, ignore_errors=True)
            df = df.to_pandas()
            logger.info("Successfully read file with Polars")
        except Exception as e:
            logger.warning(f"Polars failed, falling back to pandas: {e}")
            # Fall back to pandas if polars fails
            df = pd.read_csv(temp_file_path, na_values=MISSING_VALUES)
        
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"Column names: {list(df.columns)}")
        logger.info(f"Data types: {df.dtypes.to_dict()}")
        
        # Replace empty strings and spaces with NaN
        df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
        
        # Detect potential date columns
        date_columns = detect_date_columns(df)
        
        # Get data types of all columns
        column_types = {}
        for column in df.columns:
            column_types[column] = str(df[column].dtype)
        
        # Get basic statistics (convert to Python native types for JSON serialization)
        column_stats = {}
        for column in df.columns:
            try:
                # For numerical columns, get statistics
                if pd.api.types.is_numeric_dtype(df[column]):
                    min_val = df[column].min()
                    max_val = df[column].max()
                    mean_val = df[column].mean()
                    
                    column_stats[column] = {
                        "min": clean_for_json(min_val),
                        "max": clean_for_json(max_val),
                        "mean": clean_for_json(mean_val),
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
            except Exception as e:
                logger.warning(f"Error getting stats for column {column}: {e}")
                column_stats[column] = {
                    "missing": int(df[column].isna().sum()),
                    "missing_pct": float(df[column].isna().mean() * 100)
                }
        
        # Determine if this is likely a time series dataset
        is_time_series = len(date_columns) > 0
        
        # If no date columns found, provide helpful suggestions
        suggestions = []
        if not date_columns:
            # Check for columns that might be dates but weren't detected
            logger.warning("No automatic date detection successful. Providing manual selection options.")
            for col in df.columns:
                sample = df[col].dropna().head(10).astype(str).tolist()
                if sample:
                    suggestions.append({
                        "column": col,
                        "sample_values": sample[:5],
                        "type": str(df[col].dtype),
                        "total_values": len(df[col].dropna()),
                        "description": f"Column '{col}' with {len(sample)} non-null values"
                    })
        
        # Prepare recommendations
        recommendations = {
            "is_time_series": is_time_series,
            "date_columns": date_columns,
            "recommended_date_column": date_columns[0] if date_columns else None,
            "columns_with_missing_values": [col for col, stats in column_stats.items() 
                                           if stats.get("missing", 0) > 0],
            "suggestions": suggestions if not date_columns else [],
            "help_text": "If no date columns were detected automatically, please manually select the column containing date/time information from the list above."
        }
        
        # Prepare the final response with JSON-safe data
        response_data = {
            "success": True,
            "filename": str(file.filename),
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "column_types": {str(k): str(v) for k, v in column_types.items()},
            "column_stats": clean_for_json(column_stats),
            "date_columns": [str(col) for col in date_columns],
            "recommendations": clean_for_json(recommendations)
        }
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Error analyzing time series: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False, 
                "error": f"Error analyzing file: {str(e)}",
                "detail": "Please check that your file is a valid CSV and contains date/time data"
            }
        )
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass

@router.post("/process-time-series/")
async def process_time_series(
    file: UploadFile = File(...),
    date_column: str = Form(...),
    frequency: str = Form(...),
    drop_columns: Optional[str] = Form(None),
    imputation_method: str = Form("auto")
):
    """
    Process a time series file:
    1. Identify and convert date column
    2. Set the date column as index
    3. Impute missing values
    4. Set frequency
    5. Convert all columns to float where possible
    """
    temp_file_path = None
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
            df = pd.read_csv(temp_file_path, na_values=MISSING_VALUES)
        
        # Track processing steps for reporting
        processing_steps = []
        processing_steps.append(f"Loaded file with {len(df)} rows and {len(df.columns)} columns")
        
        # 1. Initial cleaning
        df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
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
        
        # Check if conversion was successful
        if df[date_column].isna().all():
            return JSONResponse(
                status_code=400,
                content={
                    "success": False, 
                    "error": f"Could not convert column '{date_column}' to datetime format. Please verify it contains valid date/time data."
                }
            )
        
        processing_steps.append(f"Converted '{date_column}' to datetime format")
        
        # Drop columns if specified
        if columns_to_drop:
            existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
            if existing_cols_to_drop:
                df = df.drop(columns=existing_cols_to_drop)
                processing_steps.append(f"Dropped {len(existing_cols_to_drop)} columns: {', '.join(existing_cols_to_drop)}")
        
        # 3. Set date column as index
        df = df.set_index(date_column)
        processing_steps.append(f"Set '{date_column}' as index")
        
        # 4. Handle missing values using smart imputation
        missing_before = df.isna().sum().sum()
        for col in df.columns:
            if df[col].isna().sum() > 0:
                df[col] = smart_fillna_timeseries(df[col], method=imputation_method)
        
        missing_after = df.isna().sum().sum()
        processing_steps.append(f"Imputed {missing_before - missing_after} missing values using {imputation_method} method")
        
        # 5. Set frequency
        valid_frequencies = ['D', 'B', 'W', 'M', 'Q', 'A', 'H', 'T', 'S']
        if frequency not in valid_frequencies:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": f"Invalid frequency '{frequency}'. Valid options are: {', '.join(valid_frequencies)}"}
            )
        
        # Convert to specified frequency
        try:
            df_freq = df.asfreq(frequency)
            processing_steps.append(f"Set frequency to '{frequency}'")
        except Exception as e:
            logger.warning(f"Error setting frequency: {e}")
            df_freq = df
            processing_steps.append(f"Warning: Could not set frequency to '{frequency}', keeping original index")
        
        # 6. Drop rows with all missing values (created by frequency resampling)
        freq_rows_before = len(df_freq)
        df_freq.dropna(how='all', inplace=True)
        freq_rows_dropped = freq_rows_before - len(df_freq)
        if freq_rows_dropped > 0:
            processing_steps.append(f"Dropped {freq_rows_dropped} empty rows after frequency conversion")
        
        # 7. Convert columns to appropriate numeric types
        failed_float_conversions = []
        successful_conversions = []
        
        for col in df_freq.columns:
            try:
                # First, try to clean the data
                cleaned_series = df_freq[col].replace(r',', '', regex=True)  # Remove commas
                
                # Try to convert to numeric
                numeric_series = pd.to_numeric(cleaned_series, errors='coerce')
                
                # Check if conversion was mostly successful
                conversion_rate = numeric_series.notna().sum() / len(numeric_series)
                
                if conversion_rate > 0.5:  # If more than 50% converted successfully
                    df_freq[col] = numeric_series
                    successful_conversions.append(col)
                else:
                    failed_float_conversions.append(col)
                    
            except Exception as e:
                failed_float_conversions.append(col)
                logger.warning(f"Could not convert column {col} to numeric: {e}")
        
        if successful_conversions:
            processing_steps.append(f"Successfully converted {len(successful_conversions)} columns to numeric type")
        
        if failed_float_conversions:
            processing_steps.append(f"Warning: Could not convert these columns to numeric: {', '.join(failed_float_conversions)}")
        
        # 8. Final check for any remaining missing values and clean them
        remaining_missing = df_freq.isna().sum().sum()
        if remaining_missing > 0:
            columns_with_missing = [col for col in df_freq.columns if df_freq[col].isna().sum() > 0]
            processing_steps.append(f"Warning: {remaining_missing} missing values remain in columns: {', '.join(columns_with_missing)}")
            
            # Clean remaining missing values
            for col in columns_with_missing:
                if pd.api.types.is_numeric_dtype(df_freq[col]):
                    # Fill numeric columns with 0 or median
                    fill_value = df_freq[col].median()
                    if pd.isna(fill_value):
                        fill_value = 0
                    df_freq[col] = df_freq[col].fillna(fill_value)
                else:
                    # Fill non-numeric columns with empty string
                    df_freq[col] = df_freq[col].fillna('')
        
        # 9. Clean any remaining inf values
        df_freq = df_freq.replace([np.inf, -np.inf], np.nan)
        if df_freq.isna().sum().sum() > 0:
            # Fill any new NaN values created from inf replacement
            for col in df_freq.columns:
                if df_freq[col].isna().sum() > 0:
                    if pd.api.types.is_numeric_dtype(df_freq[col]):
                        df_freq[col] = df_freq[col].fillna(0)
                    else:
                        df_freq[col] = df_freq[col].fillna('')
        
        # 10. Save the processed file
        unique_id = str(uuid.uuid4())[:8]
        output_filename = f"ts_processed_{unique_id}_{file.filename}"
        output_path = result_folder / output_filename
        
        # Reset index to include date column in the saved file
        df_output = df_freq.reset_index()
        df_output.to_csv(output_path, index=False)
        
        # Generate a report of statistics (convert to Python native types for JSON serialization)
        stats = {
            "original_rows": int(initial_rows),
            "processed_rows": int(len(df_freq)),
            "original_columns": int(len(df.columns) + 1),  # +1 for the date column that became index
            "processed_columns": int(len(df_freq.columns)),
            "frequency": str(frequency),
            "date_column": str(date_column),
            "missing_values_handled": int(missing_before - missing_after),
            "columns_with_issues": [str(col) for col in failed_float_conversions],
            "successful_numeric_conversions": [str(col) for col in successful_conversions]
        }
        
        # Prepare the final response with JSON-safe data
        response_data = {
            "success": True,
            "message": "Time series processing complete",
            "filename": str(file.filename),
            "processed_filename": str(output_filename),
            "processing_steps": [str(step) for step in processing_steps],
            "statistics": clean_for_json(stats)
        }
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Error processing time series: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Error processing time series: {str(e)}"}
        )
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass

def smart_fillna_timeseries(series: pd.Series, method='auto', rolling_window=3, interp_method='linear', fill_limit=None):
    """
    Fills NaNs in a time series with a flexible strategy based on the method specified
    """
    # Start with a copy to avoid modifying the original
    filled = series.copy()
    
    if method == 'auto':
        # Step 1: Rolling mean for short gaps
        rolling = series.rolling(window=rolling_window, min_periods=1).mean()
        filled = filled.fillna(rolling)
        
        # Step 2: Interpolation for medium gaps
        filled = filled.interpolate(method=interp_method, limit_direction='both')
        
        # Step 3: Final fallback fill
        filled = filled.fillna(method='ffill', limit=fill_limit)
        filled = filled.fillna(method='bfill', limit=fill_limit)
        
        # Step 4: If still NaN, use 0 or median
        if filled.isna().sum() > 0:
            fill_value = series.median() if not series.isna().all() else 0
            if pd.isna(fill_value):
                fill_value = 0
            filled = filled.fillna(fill_value)
        
    elif method == 'interpolate':
        filled = filled.interpolate(method='linear', limit_direction='both')
        if filled.isna().sum() > 0:
            filled = filled.fillna(0)
        
    elif method == 'mean':
        mean_val = series.mean()
        if pd.isna(mean_val):
            mean_val = 0
        filled = filled.fillna(mean_val)
        
    elif method == 'median':
        median_val = series.median()
        if pd.isna(median_val):
            median_val = 0
        filled = filled.fillna(median_val)
        
    elif method == 'ffill':
        filled = filled.fillna(method='ffill')
        if filled.isna().sum() > 0:
            filled = filled.fillna(0)
        
    elif method == 'bfill':
        filled = filled.fillna(method='bfill')
        if filled.isna().sum() > 0:
            filled = filled.fillna(0)
    
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
    temp_file_path = None
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
            df = pd.read_csv(temp_file_path, na_values=MISSING_VALUES)
        
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
                if df[column].isna().sum() > 0:
                    df[column] = df[column].fillna(0)
            
            elif method == "mean":
                mean_val = df[column].mean()
                if pd.isna(mean_val):
                    mean_val = 0
                df[column] = df[column].fillna(mean_val)
            
            elif method == "median":
                median_val = df[column].median()
                if pd.isna(median_val):
                    median_val = 0
                df[column] = df[column].fillna(median_val)
            
            elif method == "ffill":
                df[column] = df[column].fillna(method='ffill')
                if df[column].isna().sum() > 0:
                    df[column] = df[column].fillna(0)
            
            elif method == "bfill":
                df[column] = df[column].fillna(method='bfill')
                if df[column].isna().sum() > 0:
                    df[column] = df[column].fillna(0)
            
            elif method == "rolling":
                window_size = 3  # Default
                if column in column_configs:
                    window_size = column_configs[column].get('window_size', 3)
                
                rolling = df[column].rolling(window=window_size, min_periods=1).mean()
                df[column] = df[column].fillna(rolling)
                if df[column].isna().sum() > 0:
                    df[column] = df[column].fillna(0)
            
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
        
        response_data = {
            "success": True,
            "message": "Time series imputation complete",
            "filename": str(file.filename),
            "processed_filename": str(output_filename),
            "imputation_results": clean_for_json(imputation_results),
            "total_values_imputed": int(sum(result["filled_values"] for result in imputation_results.values()))
        }
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Error in custom imputation: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Error in custom imputation: {str(e)}"}
        )
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass