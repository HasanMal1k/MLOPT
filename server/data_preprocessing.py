# Import required libraries
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union, Optional
import time
import json
import uuid
import os
import re
import logging
import warnings

# Set up logging
logger = logging.getLogger('data_preprocessing')

# Define date detection patterns
DATE_PATTERNS = [
    r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
    r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
    r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
    r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
    r'\d{2}\.\d{2}\.\d{4}',  # DD.MM.YYYY
    r'\d{4}\.\d{2}\.\d{2}'   # YYYY.MM.DD
]

def detect_date_columns(df: pd.DataFrame) -> List[str]:
    """
    Intelligently detect columns containing dates
    
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
    date_name_columns = [col for col in df.columns 
                         if any(term in col.lower() 
                                for term in ['date', 'time', 'timestamp', 'year', 'month', 'day'])]
    
    # For columns with date-like names or object type, try to detect date patterns
    for col in df.select_dtypes(include=['object']).columns:
        # Skip if already identified as date
        if col in date_columns:
            continue
        
        # Check if column name suggests it's a date
        is_date_name = col in date_name_columns
        
        # Sample the first few non-null values
        sample = df[col].dropna().head(5)
        if len(sample) == 0:
            continue
        
        # Try to match patterns
        has_date_pattern = False
        for pattern in DATE_PATTERNS:
            if any(bool(re.search(pattern, str(val))) for val in sample):
                has_date_pattern = True
                break
        
        # Try converting to datetime as final check
        can_convert_to_date = False
        if has_date_pattern or is_date_name:
            # Suppress dateutil parser warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 
                                      message="Could not infer format, so each element will be parsed individually", 
                                      category=UserWarning)
                try:
                    test_conversion = pd.to_datetime(sample, errors='coerce')
                    # If at least half of the samples converted successfully
                    if test_conversion.notna().sum() >= len(sample) / 2:
                        can_convert_to_date = True
                except Exception as e:
                    logger.debug(f"Failed to test convert {col} to datetime: {e}")
        
        # Add to date columns if it passes our tests
        if (is_date_name and can_convert_to_date) or (has_date_pattern and can_convert_to_date):
            date_columns.append(col)
    
    return date_columns

def convert_date_columns(df: pd.DataFrame, date_columns: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Convert detected date columns to datetime type
    
    Args:
        df: Input dataframe
        date_columns: List of columns to convert
        
    Returns:
        Tuple of (updated dataframe, successfully converted columns)
    """
    df_result = df.copy()
    successfully_converted = []
    
    # Suppress dateutil parser warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        
        for col in date_columns:
            try:
                if col in df_result.columns:
                    df_result[col] = pd.to_datetime(df_result[col], errors='coerce')
                    successfully_converted.append(col)
            except Exception as e:
                logger.warning(f"Failed to convert {col} to datetime: {e}")
    
    return df_result, successfully_converted

def update_status(filename: str, status_folder: Path, progress: int, message: str) -> Dict:
    """
    Update the processing status of a file
    
    Args:
        filename: Name of the file being processed
        status_folder: Directory to save status files
        progress: Progress percentage (0-100)
        message: Status message
        
    Returns:
        Status dictionary
    """
    status = {
        "status": "processing" if progress < 100 else "completed",
        "progress": progress,
        "message": message,
        "timestamp": time.time()
    }
    
    status_path = status_folder / f"{filename}_status.json"
    with open(status_path, 'w') as f:
        json.dump(status, f)
    
    return status

def preprocess_file(file_path: Path, output_path: Path, 
                   status_folder: Path, progress_callback=None) -> Dict:
    """
    Main preprocessing function with improved performance and reporting
    
    Args:
        file_path: Path to the input file
        output_path: Path to save the output file
        status_folder: Path to save status files
        progress_callback: Optional callback function for progress updates
        
    Returns:
        Dictionary with preprocessing results
    """
    filename = file_path.name
    
    # 1. Read the file
    if progress_callback:
        progress_callback(10, "Reading file")
    update_status(filename, status_folder, 10, "Reading file")
    
    try:
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        else:  # Excel file
            df = pd.read_excel(file_path)
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        update_status(filename, status_folder, -1, f"Error: {str(e)}")
        return {"success": False, "error": str(e)}
    
    # Save original shape for reporting
    original_shape = df.shape
    
    # Processing results
    preprocessing_info = {
        "columns_dropped": [],
        "date_columns_detected": [],
        "columns_cleaned": [],
        "missing_value_stats": {}
    }
    
    # 2. Simple cleaning - drop columns with all missing values
    update_status(filename, status_folder, 20, "Cleaning data")
    if progress_callback:
        progress_callback(20, "Cleaning data")
    
    # Drop columns with all missing values
    null_columns = df.columns[df.isnull().all()].tolist()
    if null_columns:
        df = df.drop(columns=null_columns)
        preprocessing_info["columns_dropped"].extend(null_columns)
    
    # 3. Save basic statistics about missing values
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            missing_percentage = round((missing_count / len(df)) * 100, 2)
            preprocessing_info["missing_value_stats"][col] = {
                "missing_count": int(missing_count),
                "missing_percentage": missing_percentage,
                "imputation_method": "None"  # Default value, will be updated if imputation is applied
            }
    
    # 4. Detect and drop columns with very high missing values (e.g., > 95%)
    high_missing_cols = [col for col in df.columns 
                        if df[col].isnull().sum() > 0.95 * len(df)]
    
    if high_missing_cols:
        df = df.drop(columns=high_missing_cols)
        preprocessing_info["columns_dropped"].extend(high_missing_cols)
    
    # 5. Detect and convert date columns
    update_status(filename, status_folder, 40, "Detecting date columns")
    if progress_callback:
        progress_callback(40, "Detecting date columns")
    
    # Suppress warnings during date processing
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, 
                               message="Could not infer format, so each element will be parsed individually")
        
        date_columns = detect_date_columns(df)
        df, converted_date_columns = convert_date_columns(df, date_columns)
    
    preprocessing_info["date_columns_detected"] = converted_date_columns
    
    # 6. Handle missing values in remaining columns
    update_status(filename, status_folder, 60, "Handling missing values")
    if progress_callback:
        progress_callback(60, "Handling missing values")
    
    columns_with_missing = [col for col in df.columns if df[col].isnull().sum() > 0]
    
    for col in columns_with_missing:
        # If it's a numeric column, fill with median
        if pd.api.types.is_numeric_dtype(df[col]):
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)
            preprocessing_info["columns_cleaned"].append(col)
            if col in preprocessing_info["missing_value_stats"]:
                preprocessing_info["missing_value_stats"][col]["imputation_method"] = "median"
        
        # If it's a categorical/string column, fill with mode (most common value)
        elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            mode_value = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
            df[col] = df[col].fillna(mode_value)
            preprocessing_info["columns_cleaned"].append(col)
            if col in preprocessing_info["missing_value_stats"]:
                preprocessing_info["missing_value_stats"][col]["imputation_method"] = "mode"
        
        # For datetime columns, just leave as NaT (pandas null value for datetime)
        else:
            # No special handling needed
            pass
    
    # 7. Save the cleaned dataframe
    update_status(filename, status_folder, 80, "Saving processed file")
    if progress_callback:
        progress_callback(80, "Saving processed file")
    
    try:
        if output_path.suffix.lower() == '.csv':
            df.to_csv(output_path, index=False)
        else:  # Excel file
            df.to_excel(output_path, index=False)
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        update_status(filename, status_folder, -1, f"Error: {str(e)}")
        return {"success": False, "error": str(e)}
    
    # 8. Complete and return report
    update_status(filename, status_folder, 100, "Processing complete")
    if progress_callback:
        progress_callback(100, "Processing complete")
    
    # Final report
    report = {
        "success": True,
        "original_shape": original_shape,
        "processed_shape": df.shape,
        "preprocessing_info": preprocessing_info
    }
    
    return report

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