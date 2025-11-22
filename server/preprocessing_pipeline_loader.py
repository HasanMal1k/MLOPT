"""
Preprocessing Pipeline Loader
Loads and applies saved preprocessing pipelines to raw data for model inference
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging
import re
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# Date patterns (same as in data_preprocessing.py)
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

DATE_COLUMN_PATTERNS = [
    r"(?i)date", r"(?i)time", r"(?i)year", r"(?i)month", r"(?i)day",
    r"(?i)created", r"(?i)modified", r"(?i)timestamp", r"(?i)period"
]


def load_preprocessing_pipeline(pipeline_path: Path) -> Dict[str, Any]:
    """Load a saved preprocessing pipeline"""
    try:
        pipeline = joblib.load(pipeline_path)
        logger.info(f"Loaded preprocessing pipeline from {pipeline_path}")
        return pipeline
    except Exception as e:
        logger.error(f"Failed to load preprocessing pipeline: {e}")
        raise


def is_likely_date_column(column_name: str, sample_values: pd.Series) -> bool:
    """Check if a column is likely to contain dates"""
    # Check column name patterns
    for pattern in DATE_COLUMN_PATTERNS:
        if re.search(pattern, str(column_name)):
            return True
    
    # Check sample values
    if len(sample_values) > 0:
        sample_str = str(sample_values.iloc[0]) if not pd.isna(sample_values.iloc[0]) else ""
        for pattern in DATE_PATTERNS:
            if re.search(pattern, sample_str):
                return True
    
    return False


def apply_preprocessing_pipeline(raw_data: pd.DataFrame, pipeline: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply a saved preprocessing pipeline to raw data
    
    Args:
        raw_data: DataFrame with raw column names (as user would input)
        pipeline: Loaded preprocessing pipeline dictionary
        
    Returns:
        DataFrame with preprocessed columns (ready for model prediction)
    """
    try:
        logger.info(f"Applying preprocessing pipeline to data with shape {raw_data.shape}")
        logger.info(f"Pipeline type: {pipeline.get('pipeline_type', 'unknown')}")
        df = raw_data.copy()
        
        # Step 1: Validate input columns match expected original columns
        original_columns = pipeline.get("original_columns", [])
        if not original_columns:
            logger.warning("No original columns found in pipeline - using data as-is")
            return df
        
        missing_cols = set(original_columns) - set(df.columns)
        if missing_cols:
            logger.warning(f"Missing columns in raw data: {missing_cols}")
            # Fill missing columns with NaN
            for col in missing_cols:
                df[col] = np.nan
        
        # Add all expected columns and reorder to match training data
        for col in original_columns:
            if col not in df.columns:
                df[col] = np.nan
        
        # Reorder columns to match training data exactly
        df = df[original_columns]
        logger.info(f"Aligned columns to match training data: {df.shape}")
        
        # Step 2: Handle missing values (replace missing value indicators)
        missing_values = pipeline.get("missing_values", [])
        for col in df.columns:
            if df[col].dtype == 'object':
                # Replace string missing indicators with NaN
                df[col] = df[col].replace(missing_values, np.nan)
        
        # Step 3: Detect and parse date columns
        date_columns = pipeline.get("date_columns", [])
        for col in date_columns:
            if col in df.columns:
                try:
                    # Try to parse as datetime
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    logger.info(f"Parsed {col} as datetime")
                except Exception as e:
                    logger.warning(f"Could not parse {col} as datetime: {e}")
        
        # Step 4: Extract date features
        preprocessing_info = pipeline.get("preprocessing_info", {})
        date_features = preprocessing_info.get("date_features", {})
        
        for source_col, features in date_features.items():
            if source_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[source_col]):
                try:
                    for feature in features:
                        if feature.endswith('_year'):
                            df[feature] = df[source_col].dt.year
                        elif feature.endswith('_month'):
                            df[feature] = df[source_col].dt.month
                        elif feature.endswith('_day'):
                            df[feature] = df[source_col].dt.day
                        elif feature.endswith('_dayofweek'):
                            df[feature] = df[source_col].dt.dayofweek
                        elif feature.endswith('_quarter'):
                            df[feature] = df[source_col].dt.quarter
                    logger.info(f"Extracted date features from {source_col}")
                except Exception as e:
                    logger.warning(f"Error extracting date features from {source_col}: {e}")
        
        # Step 5: Impute missing values using saved imputation values
        categorical_modes = pipeline.get("categorical_modes", {})
        numeric_medians = pipeline.get("numeric_medians", {})
        
        # Apply categorical mode imputation
        for col, mode_val in categorical_modes.items():
            if col in df.columns and df[col].isna().sum() > 0:
                df[col] = df[col].fillna(mode_val)
                logger.info(f"Imputed {col} with mode: {mode_val}")
        
        # Apply numeric median imputation
        for col, median_val in numeric_medians.items():
            if col in df.columns and df[col].isna().sum() > 0:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(median_val)
                logger.info(f"Imputed {col} with median: {median_val}")
        
        # Step 6: Remove outliers (if applicable) - use same bounds as training
        outliers_info = preprocessing_info.get("outliers_removed", {})
        for col, info in outliers_info.items():
            if col in df.columns and 'bounds' in info:
                lower_bound = info['bounds'].get('lower')
                upper_bound = info['bounds'].get('upper')
                if lower_bound is not None and upper_bound is not None:
                    # Cap outliers instead of removing (important for single predictions)
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                    logger.info(f"Capped outliers in {col} to [{lower_bound}, {upper_bound}]")
        
        # Step 7: Apply one-hot encoding (if any categorical columns were encoded)
        one_hot_columns = pipeline.get("one_hot_columns", {})
        for source_col, derived_features in one_hot_columns.items():
            if source_col in df.columns:
                try:
                    # Create dummy variables
                    dummies = pd.get_dummies(df[source_col], prefix=source_col, drop_first=True)
                    
                    # Ensure all expected dummy columns exist
                    for feature in derived_features:
                        if feature not in dummies.columns:
                            # Column not present in this sample, add with 0
                            df[feature] = 0
                        else:
                            df[feature] = dummies[feature]
                    
                    logger.info(f"Applied one-hot encoding to {source_col}")
                except Exception as e:
                    logger.warning(f"Error in one-hot encoding {source_col}: {e}")
        
        # Step 8: Apply numeric transformations
        numeric_transformations = pipeline.get("numeric_transformations", {})
        for source_col, derived_features in numeric_transformations.items():
            if source_col in df.columns:
                try:
                    for feature in derived_features:
                        if feature.endswith('_log'):
                            df[feature] = np.log1p(df[source_col].clip(lower=0))
                        elif feature.endswith('_sqrt'):
                            df[feature] = np.sqrt(df[source_col].clip(lower=0))
                        elif feature.endswith('_squared'):
                            df[feature] = df[source_col] ** 2
                        elif feature.endswith('_reciprocal'):
                            df[feature] = 1 / df[source_col].replace([np.inf, -np.inf, 0], np.nan)
                        elif feature.endswith('_binned'):
                            # For binned features, we would need the bin edges
                            # For now, skip or use quartiles as approximation
                            pass
                    logger.info(f"Applied transformations to {source_col}")
                except Exception as e:
                    logger.warning(f"Error applying transformations to {source_col}: {e}")
        
        # Step 9: Ensure all expected final columns are present
        final_columns = []
        if "preprocessing_info" in pipeline:
            # Get all columns that should be in the final dataset
            final_columns = [col for col in df.columns]
        
        # Step 10: Fill any remaining NaN values (last resort)
        for col in df.columns:
            if df[col].isna().sum() > 0:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(0)
                else:
                    df[col] = df[col].fillna("Unknown")
        
        logger.info(f"Preprocessing complete. Output shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error applying preprocessing pipeline: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def apply_custom_transformation_pipeline(df: pd.DataFrame, pipeline: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply saved custom transformation pipeline to data
    
    Args:
        df: DataFrame to transform
        pipeline: Loaded custom transformation pipeline dictionary
        
    Returns:
        DataFrame with custom transformations applied
    """
    try:
        from transformations import apply_transformations
        
        transform_config = pipeline.get("transform_config", {})
        if not transform_config:
            logger.warning("No transformation config found in pipeline")
            return df
        
        # Apply the same transformations that were used during training
        transformed_df, _ = apply_transformations(df, transform_config)
        logger.info("Applied custom transformations from pipeline")
        
        return transformed_df
        
    except Exception as e:
        logger.error(f"Error applying custom transformation pipeline: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return df  # Return original if transformation fails


def preprocess_for_prediction(raw_data_dict: Dict[str, Any], file_id: str, processed_folder: Path) -> pd.DataFrame:
    """
    Convenience function to preprocess raw input data for a specific model
    
    Args:
        raw_data_dict: Dictionary of column_name: value pairs (raw format from user)
        file_id: The file/training ID to load the pipeline for
        processed_folder: Folder where processed files and pipelines are stored
        
    Returns:
        DataFrame with preprocessed data ready for model.predict()
    """
    # Convert raw data to DataFrame
    raw_df = pd.DataFrame([raw_data_dict])
    
    # Find the pipeline file(s)
    # Try multiple naming patterns for both auto and custom preprocessing
    possible_patterns = [
        f"processed_{file_id}_pipeline.joblib",
        f"processed_*{file_id}*_pipeline.joblib",
        f"transformed_{file_id}_pipeline.joblib",
        f"transformed_*{file_id}*_pipeline.joblib",
        f"*{file_id}*_pipeline.joblib"
    ]
    
    pipeline_paths = []
    for pattern in possible_patterns:
        matches = list(processed_folder.glob(pattern))
        pipeline_paths.extend(matches)
    
    # Also check transformed_files folder for custom transformations
    transformed_folder = Path("transformed_files")
    if transformed_folder.exists():
        for pattern in possible_patterns:
            matches = list(transformed_folder.glob(pattern))
            pipeline_paths.extend(matches)
    
    if not pipeline_paths:
        raise FileNotFoundError(f"Preprocessing pipeline not found for file_id: {file_id}")
    
    # Sort by modification time (most recent first)
    pipeline_paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    preprocessed_df = raw_df
    
    # Apply all pipelines in sequence (auto preprocessing first, then custom transformations)
    for pipeline_path in pipeline_paths:
        logger.info(f"Loading pipeline: {pipeline_path}")
        pipeline = load_preprocessing_pipeline(pipeline_path)
        
        # Check if this is a custom transformation pipeline or auto preprocessing
        if "transform_config" in pipeline:
            # Custom transformation pipeline
            preprocessed_df = apply_custom_transformation_pipeline(preprocessed_df, pipeline)
        else:
            # Auto preprocessing pipeline
            preprocessed_df = apply_preprocessing_pipeline(preprocessed_df, pipeline)
    
    return preprocessed_df
