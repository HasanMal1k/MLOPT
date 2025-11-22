import pandas as pd
import numpy as np
import polars as pl
from pathlib import Path
import logging
import time
import warnings
import traceback
import re
import os
import json
from typing import Dict, List, Tuple, Any, Optional, Union
import chardet

# Sklearn imports
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from universal_file_handler import universal_safe_read_file

import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocessing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("robust_preprocessing")

# Comprehensive list of missing value indicators
MISSING_VALUES = [
    'na', 'n/a', 'N/A', 'NAN', 'NA', 'Null', 'null', 'NULL', 
    'Nan', 'nan', 'Unknown', 'unknown', 'UNKNOWN', '-', '--', 
    '---', '----', '', ' ', '  ', '   ', '    ', '?', '??', 
    '???', '????', 'Missing', 'missing', 'MISSING', '#N/A',
    '#NA', 'None', 'none', 'NONE', 'nil', 'NIL', 'NaN', '####'
]

# Comprehensive date patterns
DATE_PATTERNS = [
    r"\b\d{4}-\d{2}-\d{2}\b",                        # YYYY-MM-DD
    r"\b\d{2}/\d{2}/\d{4}\b",                        # MM/DD/YYYY
    r"\b\d{2}-\d{2}-\d{4}\b",                        # DD-MM-YYYY
    r"\b\d{4}/\d{2}/\d{2}\b",                        # YYYY/MM/DD
    r"\b\d{2}\.\d{2}\.\d{4}\b",                      # DD.MM.YYYY
    r"\b\d{4}\.\d{2}\.\d{2}\b",                      # YYYY.MM.DD
    r"\b\d{1,2}-[A-Za-z]{3}-\d{4}\b",                # D-MMM-YYYY
    r"\b\d{1,2}\s[A-Za-z]{3,9}\s\d{4}\b",            # D Month YYYY
    r"\b[A-Za-z]{3,9}\s\d{1,2},?\s\d{4}\b",          # Month D, YYYY
    r"\b\d{1,2}[/-][A-Za-z]{3}[/-]\d{2,4}\b",        # D/MMM/YY or D-MMM-YYYY
    r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.*\b",    # ISO datetime
    r"\b\d{2}[/-]\d{2}[/-]\d{2,4}\b",                # Generic D/M/Y or M/D/Y
    r"\b\d{4}\d{2}\d{2}\b"                          # YYYYMMDD
]

# Column name patterns that suggest dates
DATE_COLUMN_PATTERNS = [
    r"(?i)date", r"(?i)time", r"(?i)year", r"(?i)month", r"(?i)day",
    r"(?i)created", r"(?i)modified", r"(?i)timestamp", r"(?i)period"
]

class ModeImputer(BaseEstimator, TransformerMixin):
    """Custom imputer that uses the mode for each column."""
    
    def __init__(self, strategy='most_frequent'):
        self.strategy = strategy
        self.modes_ = None
        
    def fit(self, X, y=None):
        """Learn the most frequent value for each column."""
        try:
            # Attempt to get mode through pandas
            if isinstance(X, pd.DataFrame):
                self.modes_ = X.mode().iloc[0]
            else:
                # Convert numpy array to pandas for mode calculation
                self.modes_ = pd.DataFrame(X).mode().iloc[0].values
            return self
        except Exception as e:
            logger.warning(f"Error in ModeImputer fit: {e}. Falling back to SimpleImputer.")
            # Fallback to sklearn SimpleImputer
            self.imputer_ = SimpleImputer(strategy=self.strategy)
            self.imputer_.fit(X)
            return self
    
    def transform(self, X):
        """Impute missing values using the modes from fit."""
        try:
            if hasattr(self, 'imputer_'):
                # Use fallback imputer
                return self.imputer_.transform(X)
            
            if isinstance(X, pd.DataFrame):
                # Handle DataFrame
                result = X.copy()
                for i, col in enumerate(X.columns):
                    result[col] = result[col].fillna(self.modes_[i])
                return result
            else:
                # Handle numpy array
                result = np.copy(X)
                mask = np.isnan(result)
                indices = np.where(mask)
                result[indices] = np.take(self.modes_, indices[1])
                return result
        except Exception as e:
            logger.error(f"Error in ModeImputer transform: {e}")
            # Return original data if transform fails
            return X

def update_status(filename: str, status_folder: Path, 
                 progress: int, message: str, 
                 results: Optional[Dict] = None) -> Dict:
    """
    Update the processing status of a file with detailed information.
    
    Args:
        filename: Name of the file being processed
        status_folder: Directory to save status files
        progress: Progress percentage (0-100)
        message: Status message
        results: Optional results to include in the status
        
    Returns:
        Status dictionary
    """
    status = {
        "status": "processing" if progress < 100 else "completed",
        "progress": progress,
        "message": message,
        "timestamp": time.time()
    }
    
    if results is not None:
        status["results"] = results
    
    status_path = status_folder / f"{filename}_status.json"
    with open(status_path, 'w') as f:
        json.dump(status, f)
    
    return status

def detect_date_columns(df: pd.DataFrame) -> List[str]:
    """
    Identify columns that likely contain date information using multiple methods.
    
    Args:
        df: Input dataframe
        
    Returns:
        List of column names that likely contain dates
    """
    date_columns = []
    
    # First check for columns that are already datetime type
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    date_columns.extend(datetime_cols)
    
    # Check for columns with date-like names
    for col in df.columns:
        # Skip if already identified
        if col in date_columns:
            continue
            
        # Check if column name suggests it's a date
        if any(re.search(pattern, col) for pattern in DATE_COLUMN_PATTERNS):
            # Sample the column to verify it contains date-like strings
            if df[col].dtype in ['object', 'string']:
                sample = df[col].dropna().astype(str).head(20).tolist()
                
                # If it has date patterns, add to list
                if any(any(re.search(pattern, str(val)) for pattern in DATE_PATTERNS) 
                       for val in sample if val is not None):
                    date_columns.append(col)
                    continue
    
    # For remaining string columns, check for date patterns directly
    string_cols = [col for col in df.select_dtypes(include=['object', 'string']).columns 
                  if col not in date_columns]
    
    for col in string_cols:
        sample = df[col].dropna().astype(str).head(20).tolist()
        pattern_matches = 0
        
        for val in sample:
            if any(re.search(pattern, str(val)) for pattern in DATE_PATTERNS):
                pattern_matches += 1
        
        # If more than 50% of sampled values match date patterns
        if pattern_matches > len(sample) * 0.5 and len(sample) > 0:
            # Confirm by attempting conversion
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    converted = pd.to_datetime(df[col].head(20), errors='coerce')
                    # If more than 50% converted successfully
                    if converted.notna().sum() > len(converted) * 0.5:
                        date_columns.append(col)
            except:
                pass
    
    return date_columns

def detect_csv_encoding(file_path: Path) -> str:
    """
    Simple encoding detection for CSV files
    """
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)  # Read first 10KB
        
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        confidence = result['confidence']
        
        logger.info(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
        
        # If confidence is low, default to utf-8
        if confidence < 0.7:
            return 'utf-8'
        
        return encoding
        
    except Exception as e:
        logger.warning(f"Encoding detection failed: {e}, using utf-8")
        return 'utf-8'

def safe_read_file(file_path: Path, missing_values: List[str] = None) -> Tuple[pd.DataFrame, bool, str]:
    """
    Enhanced version of your existing safe_read_file with encoding detection
    """
    error_msg = ""
    
    # First try with Polars for better performance
    try:
        logger.info(f"Reading {file_path} with Polars")
        df = pl.read_csv(file_path, null_values=missing_values or MISSING_VALUES, 
                         ignore_errors=True)
        return df.to_pandas(), True, ""
    except Exception as e:
        error_msg = f"Polars read failed: {e}. "
        logger.warning(f"Polars read failed for {file_path}: {e}")
    
    # Fall back to pandas with different options
    try:
        logger.info(f"Reading {file_path} with pandas")
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path, na_values=missing_values or MISSING_VALUES, 
                            keep_default_na=True, on_bad_lines='warn')
            return df, True, ""
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path, na_values=missing_values or MISSING_VALUES)
            return df, True, ""
    except Exception as e:
        error_msg += f"Pandas standard read failed: {e}. "
        logger.warning(f"Pandas standard read failed for {file_path}: {e}")
    
    # NEW: Try with encoding detection for CSV files
    if file_path.suffix.lower() == '.csv':
        try:
            detected_encoding = detect_csv_encoding(file_path)
            logger.info(f"Trying pandas with detected encoding: {detected_encoding}")
            
            df = pd.read_csv(file_path, 
                           na_values=missing_values or MISSING_VALUES,
                           keep_default_na=True, 
                           on_bad_lines='warn',
                           encoding=detected_encoding)
            return df, True, f"Read with detected encoding: {detected_encoding}"
            
        except Exception as e:
            error_msg += f"Encoding detection attempt failed: {e}. "
            logger.warning(f"Encoding detection read failed for {file_path}: {e}")
    
    # Last resort - try more permissive options (your existing fallback)
    try:
        logger.info(f"Trying permissive read for {file_path}")
        if file_path.suffix.lower() == '.csv':
            # Try multiple encodings as fallback
            encodings_to_try = ['latin1', 'cp1252', 'utf-8', 'ascii']
            
            for encoding in encodings_to_try:
                try:
                    df = pd.read_csv(file_path, 
                                   na_values=missing_values or MISSING_VALUES,
                                   keep_default_na=True, 
                                   on_bad_lines='skip', 
                                   encoding=encoding, 
                                   low_memory=False)
                    logger.info(f"Permissive read succeeded with {encoding}: {len(df)} rows")
                    return df, True, f"Used permissive reading with {encoding} (some rows may have been skipped)"
                except:
                    continue
                    
    except Exception as e:
        error_msg += f"All reading methods failed: {e}"
        logger.error(f"All methods failed to read {file_path}: {e}")
        return pd.DataFrame(), False, error_msg
    
    return pd.DataFrame(), False, "Unknown error reading file"

def preprocess_file(file_path: Path, output_path: Path, 
                   status_folder: Path, progress_callback=None) -> Dict:
    """
    Enhanced preprocessing function with robust error handling and comprehensive cleaning.
    
    Args:
        file_path: Path to the input file
        output_path: Path to save the output file
        status_folder: Path to save status files
        progress_callback: Optional callback function for progress updates
        
    Returns:
        Dictionary with preprocessing results
    """
    filename = file_path.name
    logger.info(f"Starting preprocessing of {filename}")
    
    try:
        # Setup preprocessing info structure
        preprocessing_info = {
            "columns_dropped": [],
            "date_columns_detected": [],
            "columns_cleaned": [],
            "missing_value_stats": {},
            "dropped_by_unique_value": [],
            "engineered_features": [],
            "transformation_details": {}
        }
        
        # 1. Read the file
        update_status(filename, status_folder, 5, "Reading file")
        if progress_callback:
            progress_callback(5, "Reading file")
        
        df, read_success, read_message = safe_read_file(file_path, MISSING_VALUES)
        
        if not read_success or df is None or df.empty:
            error_msg = f"Failed to read file: {read_message}"
            logger.error(error_msg)
            update_status(filename, status_folder, -1, error_msg)
            return {"success": False, "error": error_msg}
        
        # Save original shape and columns for pipeline
        original_shape = df.shape
        original_columns = df.columns.tolist()  # Store for pipeline
        logger.info(f"Original shape: {original_shape}")
        logger.info(f"Original columns: {original_columns}")
        
        if original_shape[0] == 0 or original_shape[1] == 0:
            error_msg = f"File is empty or has no columns: {original_shape}"
            logger.error(error_msg)
            update_status(filename, status_folder, -1, error_msg)
            return {"success": False, "error": error_msg}
        
        # 2. Handle whitespace and make a copy for safety
        update_status(filename, status_folder, 10, "Initial cleaning")
        if progress_callback:
            progress_callback(10, "Initial cleaning")
        
        try:
            # Remove whitespace in string cells
            df = df.replace(r'^\s*$', np.nan, regex=True)
            
            # Drop rows with all missing values
            initial_row_count = len(df)
            df.dropna(how='all', inplace=True)
            dropped_rows = initial_row_count - len(df)
            
            if dropped_rows > 0:
                logger.info(f"Dropped {dropped_rows} rows that were completely empty")
        except Exception as e:
            logger.warning(f"Error in initial cleaning: {e}")
            # Continue despite errors in this step
        
        # 3. Gather statistics about missing values
        update_status(filename, status_folder, 15, "Analyzing missing values")
        if progress_callback:
            progress_callback(15, "Analyzing missing values")
        
        try:
            for col in df.columns:
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    missing_percentage = round((missing_count / len(df)) * 100, 2)
                    preprocessing_info["missing_value_stats"][col] = {
                        "missing_count": int(missing_count),
                        "missing_percentage": missing_percentage,
                        "imputation_method": "None"  # Will be updated later
                    }
            logger.info(f"Found {len(preprocessing_info['missing_value_stats'])} columns with missing values")
        except Exception as e:
            logger.warning(f"Error analyzing missing values: {e}")
        
        # 4. Drop problematic columns
        update_status(filename, status_folder, 20, "Removing problematic columns")
        if progress_callback:
            progress_callback(20, "Removing problematic columns")
        
        try:
            # Drop duplicate column names (retaining first)
            if len(df.columns) != len(df.columns.unique()):
                df = df.loc[:, ~df.columns.duplicated()]
                logger.info("Removed duplicate column names")
            
            # Drop columns with extremely high missing values (>95%)
            high_missing_cols = [col for col in df.columns 
                                if df[col].isna().sum() > 0.95 * len(df)]
            
            if high_missing_cols:
                df = df.drop(columns=high_missing_cols)
                preprocessing_info["columns_dropped"].extend(high_missing_cols)
                logger.info(f"Dropped {len(high_missing_cols)} columns with >95% missing values")
        except Exception as e:
            logger.warning(f"Error removing problematic columns: {e}")
        
        # 5. Detect and convert date columns
        update_status(filename, status_folder, 30, "Processing date columns")
        if progress_callback:
            progress_callback(30, "Processing date columns")
        
        try:
            date_columns = detect_date_columns(df)
            
            # Convert detected date columns
            for col in date_columns:
                if col in df.columns:  # Safety check
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        df[col] = pd.to_datetime(df[col], errors='coerce')
            
            preprocessing_info["date_columns_detected"] = date_columns
            logger.info(f"Detected and converted {len(date_columns)} date columns")
            
            # Basic feature engineering for date columns
            if date_columns:
                datetime_features = []
                for col in date_columns:
                    if col in df.columns and df[col].dtype == 'datetime64[ns]':
                        try:
                            # Extract basic date components
                            df[f'{col}_year'] = df[col].dt.year
                            df[f'{col}_month'] = df[col].dt.month
                            df[f'{col}_day'] = df[col].dt.day
                            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
                            df[f'{col}_quarter'] = df[col].dt.quarter
                            
                            new_features = [f'{col}_year', f'{col}_month', f'{col}_day', 
                                          f'{col}_dayofweek', f'{col}_quarter']
                            preprocessing_info["engineered_features"].extend(new_features)
                            
                            datetime_features.append({
                                "source_column": col,
                                "derived_features": new_features
                            })
                            
                            logger.info(f"Created {len(new_features)} date features from {col}")
                        except Exception as e:
                            logger.warning(f"Error creating date features for {col}: {e}")
                
                if datetime_features:
                    preprocessing_info["transformation_details"]["datetime_features"] = datetime_features
            
        except Exception as e:
            logger.warning(f"Error processing date columns: {e}")
        
        # 6. Prepare for imputation
        update_status(filename, status_folder, 40, "Preparing for imputation")
        if progress_callback:
            progress_callback(40, "Preparing for imputation")
        
        try:
            # Identify column types for imputation
            nominal_columns = [col for col in df.columns 
                              if col not in date_columns and 
                              df[col].dtype in ['object', 'category', 'string', 'bool']]
            
            non_nominal_columns = [col for col in df.columns 
                                  if col not in nominal_columns and 
                                  col not in date_columns and
                                  not col.endswith(('_year', '_month', '_day', '_dayofweek', '_quarter'))]
            
            logger.info(f"Column types: {len(nominal_columns)} categorical, " 
                       f"{len(non_nominal_columns)} numeric, {len(date_columns)} date")
        except Exception as e:
            logger.warning(f"Error preparing for imputation: {e}")
            # Define in case of error
            nominal_columns = []
            non_nominal_columns = []
        
        # 7. Handle missing values with advanced techniques
        update_status(filename, status_folder, 50, "Handling missing values")
        if progress_callback:
            progress_callback(50, "Handling missing values")
        
        try:
            # Create a copy of the dataframe for safety during imputation
            df_copy = df.copy()
            
            # Initialize pipeline storage variables at function scope
            categorical_modes = {}  # Store modes for pipeline
            numeric_medians = {}    # Store medians for pipeline
            
            # For categorical columns
            if nominal_columns:
                for col in nominal_columns:
                    if col in preprocessing_info["missing_value_stats"]:
                        if df_copy[col].isna().sum() > 0:
                            try:
                                # Get the mode for this column
                                mode_val = df_copy[col].mode().iloc[0] if not df_copy[col].mode().empty else "Unknown"
                                categorical_modes[col] = mode_val  # Store for pipeline
                                df_copy[col] = df_copy[col].fillna(mode_val)
                                preprocessing_info["columns_cleaned"].append(col)
                                preprocessing_info["missing_value_stats"][col]["imputation_method"] = "mode"
                                preprocessing_info["missing_value_stats"][col]["imputation_value"] = str(mode_val)
                            except Exception as e:
                                logger.warning(f"Error imputing column {col}: {e}")
            
            # For numeric columns
            if non_nominal_columns:
                try:
                    # Try using KNN imputation for numeric columns
                    missing_in_numeric = any(df_copy[col].isna().sum() > 0 for col in non_nominal_columns)
                    
                    if missing_in_numeric and len(df_copy) > 10 and len(non_nominal_columns) > 0:
                        # Determine optimal k for KNN (smaller of 5 or half the dataset)
                        k = min(5, max(2, len(df_copy) // 2))
                        
                        # Prepare data for KNN imputation
                        numeric_data = df_copy[non_nominal_columns].copy()
                        
                        # Convert to numeric where possible
                        for col in non_nominal_columns:
                            try:
                                numeric_data[col] = pd.to_numeric(numeric_data[col], errors='coerce')
                            except:
                                pass
                        
                        # Apply KNN imputation
                        from sklearn.impute import KNNImputer
                        imputer = KNNImputer(n_neighbors=k)
                        imputed_values = imputer.fit_transform(numeric_data)
                        
                        # Update the dataframe with imputed values
                        for i, col in enumerate(non_nominal_columns):
                            if col in preprocessing_info["missing_value_stats"]:
                                df_copy[col] = imputed_values[:, i]
                                preprocessing_info["columns_cleaned"].append(col)
                                preprocessing_info["missing_value_stats"][col]["imputation_method"] = "KNN"
                    
                    else:
                        # Fall back to simpler imputation if KNN doesn't make sense
                        for col in non_nominal_columns:
                            if col in preprocessing_info["missing_value_stats"] and df_copy[col].isna().sum() > 0:
                                # Use median for imputation
                                median_val = df_copy[col].median()
                                numeric_medians[col] = float(median_val) if not pd.isna(median_val) else 0.0
                                df_copy[col] = df_copy[col].fillna(median_val)
                                preprocessing_info["columns_cleaned"].append(col)
                                preprocessing_info["missing_value_stats"][col]["imputation_method"] = "median"
                                preprocessing_info["missing_value_stats"][col]["imputation_value"] = float(median_val) if not pd.isna(median_val) else 0.0
                
                except Exception as e:
                    logger.warning(f"Error in KNN imputation: {e}, falling back to median")
                    
                    # Fall back to median imputation for numeric columns
                    for col in non_nominal_columns:
                        if col in preprocessing_info["missing_value_stats"] and df_copy[col].isna().sum() > 0:
                            try:
                                median_val = df_copy[col].median()
                                numeric_medians[col] = float(median_val) if not pd.isna(median_val) else 0.0
                                df_copy[col] = df_copy[col].fillna(median_val)
                                preprocessing_info["columns_cleaned"].append(col)
                                preprocessing_info["missing_value_stats"][col]["imputation_method"] = "median"
                                preprocessing_info["missing_value_stats"][col]["imputation_value"] = float(median_val) if not pd.isna(median_val) else 0.0
                            except Exception as e:
                                logger.warning(f"Error with median imputation for {col}: {e}")
            
            # Update the main dataframe after imputation
            df = df_copy
            
            logger.info(f"Cleaned {len(preprocessing_info['columns_cleaned'])} columns with missing values")
        except Exception as e:
            logger.error(f"Error handling missing values: {e}")
            # Continue despite errors, using the original dataframe
        
        # 8. Remove columns with single value (no variance)
        update_status(filename, status_folder, 70, "Final cleaning")
        if progress_callback:
            progress_callback(70, "Final cleaning")
        
        try:
            single_value_cols = [col for col in df.columns if df[col].nunique() <= 1]
            if single_value_cols:
                df = df.drop(columns=single_value_cols)
                preprocessing_info["dropped_by_unique_value"] = single_value_cols
                logger.info(f"Dropped {len(single_value_cols)} columns with only a single value")
        except Exception as e:
            logger.warning(f"Error removing single-value columns: {e}")
        
        # 9. Basic feature engineering for numeric columns
        update_status(filename, status_folder, 75, "Feature engineering")
        if progress_callback:
            progress_callback(75, "Feature engineering")
        
        try:
            numeric_transformations = []
            categorical_encodings = []
            
            # Get current numeric columns (excluding engineered date features)
            current_numeric_cols = [col for col in df.select_dtypes(include=['number']).columns
                                  if not col.endswith(('_year', '_month', '_day', '_dayofweek', '_quarter'))]
            
            # Apply basic transformations to highly skewed numeric columns
            for col in current_numeric_cols:
                if df[col].nunique() > 2:  # Skip binary columns
                    try:
                        skewness = df[col].skew()
                        derived_features = []
                        
                        # Log transformation for highly positively skewed data
                        if skewness > 2 and (df[col] > 0).all():
                            df[f'{col}_log'] = np.log1p(df[col])
                            derived_features.append(f'{col}_log')
                            preprocessing_info["engineered_features"].append(f'{col}_log')
                        
                        # Square root transformation for moderately skewed data
                        if skewness > 1 and (df[col] >= 0).all():
                            df[f'{col}_sqrt'] = np.sqrt(df[col])
                            derived_features.append(f'{col}_sqrt')
                            preprocessing_info["engineered_features"].append(f'{col}_sqrt')
                        
                        if derived_features:
                            numeric_transformations.append({
                                "source_column": col,
                                "derived_features": derived_features,
                                "skew": float(skewness)
                            })
                    
                    except Exception as e:
                        logger.warning(f"Error in feature engineering for {col}: {e}")
            
            # One-hot encoding for low-cardinality categorical columns
            categorical_cols = [col for col in df.select_dtypes(include=['object', 'category']).columns]
            for col in categorical_cols:
                if 2 <= df[col].nunique() <= 10:  # Only encode columns with reasonable cardinality
                    try:
                        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                        df = pd.concat([df, dummies], axis=1)
                        
                        derived_features = dummies.columns.tolist()
                        preprocessing_info["engineered_features"].extend(derived_features)
                        
                        categorical_encodings.append({
                            "source_column": col,
                            "encoding_type": "one_hot",
                            "derived_features": derived_features,
                            "cardinality": int(df[col].nunique())
                        })
                    except Exception as e:
                        logger.warning(f"Error in one-hot encoding for {col}: {e}")
            
            # Store transformation details
            if numeric_transformations:
                preprocessing_info["transformation_details"]["numeric_transformations"] = numeric_transformations
            if categorical_encodings:
                preprocessing_info["transformation_details"]["categorical_encodings"] = categorical_encodings
            
            logger.info(f"Created {len(preprocessing_info['engineered_features'])} engineered features")
            
        except Exception as e:
            logger.warning(f"Error in feature engineering: {e}")
        
        # 10. Verify data quality and ensure no missing values remain
        update_status(filename, status_folder, 80, "Final quality check")
        if progress_callback:
            progress_callback(80, "Final quality check")
        
        try:
            # Check for any remaining missing values
            remaining_missing = df.isna().sum().sum()
            
            if remaining_missing > 0:
                logger.warning(f"There are still {remaining_missing} missing values after processing")
                
                # Last resort fill with meaningful constants
                for col in df.columns:
                    if df[col].isna().sum() > 0:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            df[col] = df[col].fillna(0)
                        else:
                            df[col] = df[col].fillna("Unknown")
                        
                        if col not in preprocessing_info["columns_cleaned"]:
                            preprocessing_info["columns_cleaned"].append(col)
            
            # Ensure we still have data
            if len(df) == 0 or len(df.columns) == 0:
                error_msg = f"No data remains after preprocessing. Original: {original_shape}, Current: {df.shape}"
                logger.error(error_msg)
                update_status(filename, status_folder, -1, error_msg)
                return {"success": False, "error": error_msg}
        except Exception as e:
            logger.warning(f"Error in final quality check: {e}")
        
        # 11. Save the processed file
        update_status(filename, status_folder, 90, "Saving processed file")
        if progress_callback:
            progress_callback(90, "Saving processed file")
        
        try:
            # Create a directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to the correct format
            if output_path.suffix.lower() == '.csv':
                df.to_csv(output_path, index=False)
            elif output_path.suffix.lower() in ['.xlsx', '.xls']:
                df.to_excel(output_path, index=False)
            else:
                # Default to CSV if extension is not recognized
                output_csv = output_path.with_suffix('.csv')
                df.to_csv(output_csv, index=False)
                logger.info(f"Unrecognized extension, saved as CSV: {output_csv}")
            
            logger.info(f"Successfully saved processed file to {output_path}")
        except Exception as e:
            error_msg = f"Error saving processed file: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            update_status(filename, status_folder, -1, error_msg)
            return {"success": False, "error": error_msg}
        
        # 12. Complete processing and return report
        final_shape = df.shape
        processed_percentage = round((len(preprocessing_info["columns_cleaned"]) / original_shape[1]) * 100, 1) if original_shape[1] > 0 else 0
        
        logger.info(f"Preprocessing complete. Original shape: {original_shape}, Final shape: {final_shape}")
        logger.info(f"Processed {len(preprocessing_info['columns_cleaned'])} columns ({processed_percentage}%)")
        
        # Save comprehensive preprocessing pipeline for reproducible predictions
        try:
            import joblib
            pipeline_path = output_path.with_name(f"{output_path.stem}_pipeline.joblib")
            
            # Build comprehensive pipeline with all transformation details
            preprocessing_pipeline = {
                # Metadata
                "pipeline_type": "auto_preprocessing",
                "missing_values": MISSING_VALUES,
                "date_patterns": DATE_PATTERNS,
                "preprocessing_info": preprocessing_info,
                "original_shape": list(original_shape),
                "final_shape": list(final_shape),
                "original_columns": original_columns,
                
                # Column type information
                "date_columns": date_columns if 'date_columns' in locals() else [],
                "nominal_columns": nominal_columns if 'nominal_columns' in locals() else [],
                "non_nominal_columns": non_nominal_columns if 'non_nominal_columns' in locals() else [],
                
                # Imputation values for inference
                "categorical_modes": categorical_modes if 'categorical_modes' in locals() else {},
                "numeric_medians": numeric_medians if 'numeric_medians' in locals() else {},
                
                # Feature engineering mappings
                "one_hot_columns": {},
                "numeric_transformations": {},
                "date_feature_mappings": {}
            }
            
            # Extract one-hot encoding details from preprocessing_info
            transformation_details = preprocessing_info.get("transformation_details", {})
            if isinstance(transformation_details, dict):
                # Store one-hot encoding mappings
                categorical_encodings = transformation_details.get("categorical_encodings", [])
                if isinstance(categorical_encodings, list):
                    for encoding in categorical_encodings:
                        if isinstance(encoding, dict):
                            source_col = encoding.get("source_column")
                            derived_features = encoding.get("derived_features", [])
                            if source_col:
                                preprocessing_pipeline["one_hot_columns"][source_col] = derived_features
                
                # Store numeric transformation details
                numeric_transformations = transformation_details.get("numeric_transformations", [])
                if isinstance(numeric_transformations, list):
                    for transform in numeric_transformations:
                        if isinstance(transform, dict):
                            source_col = transform.get("source_column")
                            derived_features = transform.get("derived_features", [])
                            if source_col:
                                preprocessing_pipeline["numeric_transformations"][source_col] = derived_features
                
                # Store date feature mappings
                datetime_features = transformation_details.get("datetime_features", [])
                if isinstance(datetime_features, list):
                    for date_feat in datetime_features:
                        if isinstance(date_feat, dict):
                            source_col = date_feat.get("source_column")
                            derived_features = date_feat.get("derived_features", [])
                            if source_col:
                                preprocessing_pipeline["date_feature_mappings"][source_col] = derived_features
            
            # Save the pipeline
            joblib.dump(preprocessing_pipeline, pipeline_path)
            logger.info(f"Saved comprehensive preprocessing pipeline to {pipeline_path}")
            logger.info(f"Pipeline contains: {len(original_columns)} original columns, "
                       f"{len(preprocessing_pipeline['categorical_modes'])} categorical modes, "
                       f"{len(preprocessing_pipeline['numeric_medians'])} numeric medians, "
                       f"{len(preprocessing_pipeline['one_hot_columns'])} one-hot encodings")
            
        except Exception as e:
            logger.warning(f"Could not save preprocessing pipeline: {e}")
            import traceback
            logger.warning(traceback.format_exc())
        
        # Final report
        report = {
            "success": True,
            "original_shape": list(original_shape),
            "processed_shape": list(final_shape),
            "preprocessing_info": preprocessing_info,
            "message": f"Successfully processed {filename}. Cleaned {len(preprocessing_info['columns_cleaned'])} columns."
        }
        
        update_status(filename, status_folder, 100, "Processing complete", report)
        if progress_callback:
            progress_callback(100, "Processing complete")
        
        return report
        
    except Exception as e:
        error_msg = f"Unexpected error during preprocessing: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        
        # Update status to indicate error
        update_status(filename, status_folder, -1, error_msg)
        
        return {
            "success": False, 
            "error": error_msg,
            "original_shape": [0, 0],
            "processed_shape": [0, 0],
            "preprocessing_info": {
                "columns_dropped": [],
                "date_columns_detected": [],
                "columns_cleaned": [],
                "missing_value_stats": {},
                "dropped_by_unique_value": [],
                "engineered_features": [],
                "transformation_details": {}
            }
        }
        
def generate_eda_report(df: pd.DataFrame) -> str:
    """
    Generate an EDA report for a dataframe
    
    Args:
        df: Input dataframe
        
    Returns:
        HTML report as a string
    """
    try:
        # Try to use ydata-profiling if available
        from ydata_profiling import ProfileReport
        
        # Generate report
        profile = ProfileReport(df, title="Data Profiling Report", minimal=True)
        return profile.to_html()
    except:
        # Fallback to simple report
        return generate_simple_eda_report(df)

def generate_simple_eda_report(df: pd.DataFrame) -> str:
    """
    Generate a simple EDA report when ydata-profiling fails
    
    Args:
        df: Input dataframe
        
    Returns:
        HTML report as a string
    """
    # Create basic statistics
    info = {
        "rows": len(df),
        "columns": len(df.columns),
        "dtypes": df.dtypes.apply(lambda x: str(x)).to_dict(),
        "missing_values": df.isna().sum().to_dict(),
        "unique_values": {col: df[col].nunique() for col in df.columns},
    }
    
    # Create numeric summaries where applicable
    numeric_summaries = {}
    for col in df.select_dtypes(include=['number']).columns:
        numeric_summaries[col] = {
            "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
            "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
            "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
            "median": float(df[col].median()) if not pd.isna(df[col].median()) else None,
            "std": float(df[col].std()) if not pd.isna(df[col].std()) else None,
        }
    
    # Generate HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Simple Data Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .summary {{ display: flex; gap: 20px; margin-bottom: 20px; }}
            .summary-box {{ padding: 15px; background-color: #f8f9fa; border-radius: 4px; border: 1px solid #dee2e6; flex: 1; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Data Analysis Report</h1>
            
            <div class="summary">
                <div class="summary-box">
                    <h3>Rows</h3>
                    <p>{info['rows']}</p>
                </div>
                <div class="summary-box">
                    <h3>Columns</h3>
                    <p>{info['columns']}</p>
                </div>
            </div>
            
            <h2>Column Overview</h2>
            <table>
                <tr>
                    <th>Column</th>
                    <th>Type</th>
                    <th>Missing Values</th>
                    <th>Missing %</th>
                    <th>Unique Values</th>
                </tr>
    """
    
    # Add rows for each column
    for col in df.columns:
        missing = info['missing_values'].get(col, 0)
        missing_percent = round((missing / info['rows']) * 100, 2) if info['rows'] > 0 else 0
        
        html += f"""
                <tr>
                    <td>{col}</td>
                    <td>{info['dtypes'].get(col, 'unknown')}</td>
                    <td>{missing}</td>
                    <td>{missing_percent}%</td>
                    <td>{info['unique_values'].get(col, 'N/A')}</td>
                </tr>
        """
    
    html += """
            </table>
            
            <h2>Numeric Column Statistics</h2>
    """
    
    if numeric_summaries:
        html += """
            <table>
                <tr>
                    <th>Column</th>
                    <th>Min</th>
                    <th>Max</th>
                    <th>Mean</th>
                    <th>Median</th>
                    <th>Std. Deviation</th>
                </tr>
        """
        
        for col, stats in numeric_summaries.items():
            html += f"""
                <tr>
                    <td>{col}</td>
                    <td>{stats['min']}</td>
                    <td>{stats['max']}</td>
                    <td>{stats['mean']:.2f if stats['mean'] is not None else 'N/A'}</td>
                    <td>{stats['median']:.2f if stats['median'] is not None else 'N/A'}</td>
                    <td>{stats['std']:.2f if stats['std'] is not None else 'N/A'}</td>
                </tr>
            """
        
        html += """
            </table>
        """
    else:
        html += """
            <p>No numeric columns found in the dataset.</p>
        """
    
    # Add sample data
    html += f"""
            <h2>Sample Data (First 10 Rows)</h2>
            <table>
                <tr>
    """
    
    # Add header row
    for col in df.columns:
        html += f"<th>{col}</th>"
    
    html += """
                </tr>
    """
    
    # Add sample data rows
    for _, row in df.head(10).iterrows():
        html += "<tr>"
        for col in df.columns:
            value = row[col]
            # Format value for display
            if pd.isna(value):
                display_val = "NA"
            elif isinstance(value, float):
                display_val = f"{value:.4f}"
            else:
                display_val = str(value)
            
            html += f"<td>{display_val}</td>"
        html += "</tr>"
    
    html += """
            </table>
        </div>
    </body>
    </html>
    """
    
    return html

def process_data_file(input_file: Union[str, Path], 
                     output_dir: Union[str, Path],
                     status_dir: Union[str, Path],
                     progress_callback=None) -> Dict:
    """
    Process a data file with complete error handling and reporting
    
    Args:
        input_file: Path to the input file
        output_dir: Directory to save output files
        status_dir: Directory to save status files
        progress_callback: Optional callback function for progress updates
        
    Returns:
        Dictionary with processing results and report
    """
    # Convert to Path objects
    input_path = Path(input_file)
    output_dir = Path(output_dir)
    status_dir = Path(status_dir)
    
    # Ensure directories exist
    output_dir.mkdir(exist_ok=True, parents=True)
    status_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate output file path
    output_path = output_dir / f"processed_{input_path.name}"
    
    # Process the file
    result = preprocess_file(input_path, output_path, status_dir, progress_callback)
    
    # Return the results
    return result