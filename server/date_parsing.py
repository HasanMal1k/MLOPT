"""
Enhanced date parsing utilities for data_preprocessing.py
This module specifically addresses pandas date parsing warnings.
"""

import pandas as pd
import numpy as np
import re
from typing import List, Union, Optional, Tuple
import warnings
import logging

logger = logging.getLogger('date_parsing')

# Common date formats to try
COMMON_DATE_FORMATS = [
    # Standard formats
    '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d',
    '%m-%d-%Y', '%d-%m-%Y', '%Y.%m.%d', '%d.%m.%Y',
    # With time
    '%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M:%S', '%d/%m/%Y %H:%M:%S',
    '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S.%f',
    # Month names
    '%b %d, %Y', '%B %d, %Y', '%d %b %Y', '%d %B %Y',
    # Short years
    '%m/%d/%y', '%d/%m/%y', '%y/%m/%d',
    # Other common formats
    '%d-%b-%Y', '%Y%m%d'
]

def detect_date_format(date_sample: List[str]) -> Optional[str]:
    """
    Detect the date format from a sample of dates.
    
    Args:
        date_sample: List of date strings to analyze
        
    Returns:
        Detected date format string, or None if no common format found
    """
    # Clean the sample by removing empty values and limiting to a reasonable size
    clean_sample = [str(x) for x in date_sample if x and pd.notna(x)][:20]
    
    if not clean_sample:
        return None
    
    # Try common formats to see which one works for all samples
    format_counts = {}
    
    for date_format in COMMON_DATE_FORMATS:
        success_count = 0
        
        for date_str in clean_sample:
            try:
                pd.to_datetime(date_str, format=date_format)
                success_count += 1
            except:
                # This format doesn't work for this string
                pass
        
        if success_count > 0:
            format_counts[date_format] = success_count
    
    # If any format worked for the majority of samples, use it
    if format_counts:
        best_format = max(format_counts.items(), key=lambda x: x[1])
        coverage = best_format[1] / len(clean_sample)
        
        if coverage >= 0.5:  # If the format works for at least 50% of samples
            return best_format[0]
    
    # No common format found
    return None


def parse_dates_with_format_detection(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Parse date columns with automatic format detection to avoid warnings.
    
    Args:
        df: Input dataframe
        columns: List of columns to parse as dates
        
    Returns:
        Tuple of (updated dataframe, list of successfully converted columns)
    """
    # Suppress dateutil parser warnings temporarily
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, 
                               message="Could not infer format, so each element will be parsed individually")
        
        successfully_converted = []
        
        for col in columns:
            if col not in df.columns:
                continue
            
            # Get a sample of non-null values
            sample = df[col].dropna().astype(str).unique()[:20].tolist()
            
            if not sample:
                continue
            
            # Try to detect date format
            date_format = detect_date_format(sample)
            
            try:
                if date_format:
                    # Use the detected format
                    df[col] = pd.to_datetime(df[col], format=date_format, errors='coerce')
                    logger.info(f"Converted column '{col}' to datetime using format: {date_format}")
                else:
                    # Fall back to automatic parsing but suppress warnings
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    logger.info(f"Converted column '{col}' to datetime using automatic format detection")
                
                successfully_converted.append(col)
            except Exception as e:
                logger.warning(f"Failed to convert column '{col}' to datetime: {e}")
                continue
    
    return df, successfully_converted


def parse_dates_from_mixed_formats(series: pd.Series) -> pd.Series:
    """
    Convert a pandas Series to datetime, trying multiple formats and handling mixed formats.
    
    Args:
        series: Input series with date strings
        
    Returns:
        Series converted to datetime, with unparseable values as NaT
    """
    if series.empty:
        return series
    
    # Initial series of NaT values with the same index
    result = pd.Series([pd.NaT] * len(series), index=series.index)
    
    # Suppress dateutil parser warnings temporarily
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, 
                              message="Could not infer format, so each element will be parsed individually")
        
        # Try all formats for each element
        for date_format in COMMON_DATE_FORMATS:
            # Get mask of values still NaT in result
            mask = result.isna()
            
            if not mask.any():
                break  # All values have been parsed successfully
            
            # Get the items still to be parsed
            to_parse = series[mask]
            
            # Try current format for all remaining items
            try:
                parsed = pd.to_datetime(to_parse, format=date_format, errors='coerce')
                
                # Update only the values that were successfully parsed
                valid_mask = ~parsed.isna()
                if valid_mask.any():
                    result[mask & mask.index.isin(parsed[valid_mask].index)] = parsed[valid_mask]
            except:
                continue
    
    # For any remaining unparsed values, try the flexible parser
    mask = result.isna()
    if mask.any():
        to_parse = series[mask]
        try:
            parsed = pd.to_datetime(to_parse, errors='coerce')
            # Update the result
            result[mask] = parsed
        except:
            pass
    
    return result


def convert_date_columns_improved(df: pd.DataFrame, date_columns: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Improved version of convert_date_columns that handles mixed formats and suppresses warnings.
    
    Args:
        df: Input dataframe
        date_columns: List of columns to convert
        
    Returns:
        Tuple of (updated dataframe, successfully converted columns)
    """
    df_result = df.copy()
    successfully_converted = []
    
    # Suppress dateutil parser warnings temporarily
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        
        for col in date_columns:
            if col not in df_result.columns:
                continue
            
            try:
                # First try an optimized approach using format detection
                df_result[col] = parse_dates_from_mixed_formats(df_result[col])
                
                # Check if conversion was successful (at least some values were converted)
                if df_result[col].dtype == 'datetime64[ns]' and not df_result[col].isna().all():
                    successfully_converted.append(col)
                else:
                    logger.warning(f"Column {col} couldn't be converted to datetime (all values became NaT)")
            except Exception as e:
                logger.warning(f"Failed to convert {col} to datetime: {e}")
    
    return df_result, successfully_converted