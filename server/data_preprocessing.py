# Add these imports to data_preprocessing.py
import warnings
from date_parsing import convert_date_columns_improved, parse_dates_with_format_detection

# Replace the existing detect_date_columns function with this improved version
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

# Replace the existing convert_date_columns function with this fixed version
def convert_date_columns(df: pd.DataFrame, date_columns: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Convert detected date columns to datetime type
    
    Args:
        df: Input dataframe
        date_columns: List of columns to convert
        
    Returns:
        Tuple of (updated dataframe, successfully converted columns)
    """
    # Use the improved date parsing function that suppresses warnings
    return convert_date_columns_improved(df, date_columns)

# Update the preprocess_file function to use the improved date parsing
# Replace the date column detection and conversion section with:

def preprocess_file(file_path: Path, output_path: Path, 
                   status_folder: Path, progress_callback=None) -> Dict:
    """Main preprocessing function with improved performance and reporting"""
    # ... (keep existing code until the date detection section)
    
    # 5. Detect and convert date columns
    update_status(filename, status_folder, 40, "Detecting date columns")
    if progress_callback:
        progress_callback(40, "Detecting date columns")
    
    # Suppress warnings during date processing
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, 
                               message="Could not infer format, so each element will be parsed individually")
        
        date_columns = detect_date_columns(df)
        df, converted_date_columns = convert_date_columns_improved(df, date_columns)
    
    # ... (continue with the rest of the function)
    
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
            .tab {{ overflow: hidden; border: 1px solid #ccc; background-color: #f1f1f1; }}
            .tab button {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; }}
            .tab button:hover {{ background-color: #ddd; }}
            .tab button.active {{ background-color: #ccc; }}
            .tabcontent {{ display: none; padding: 6px 12px; border: 1px solid #ccc; border-top: none; }}
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


def analyze_dataframe_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze the structure of a dataframe including data types and distributions
    
    Args:
        df: Input dataframe
        
    Returns:
        Dictionary with structure analysis
    """
    # Basic dataframe info
    info = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "memory_usage": df.memory_usage(deep=True).sum() / (1024 * 1024),  # MB
    }
    
    # Column type stats
    column_types = df.dtypes.astype(str).value_counts().to_dict()
    info["column_types"] = {str(k): int(v) for k, v in column_types.items()}
    
    # Missing value analysis
    missing_counts = df.isna().sum()
    missing_percentages = (missing_counts / len(df) * 100).round(2)
    
    columns_with_missing = []
    for col, count in missing_counts.items():
        if count > 0:
            columns_with_missing.append({
                "name": col,
                "missing_count": int(count),
                "missing_percentage": float(missing_percentages[col])
            })
    
    info["missing_values"] = {
        "total_missing_cells": int(df.isna().sum().sum()),
        "columns_with_missing": columns_with_missing,
        "columns_with_missing_count": len(columns_with_missing)
    }
    
    # Duplicate rows analysis
    duplicate_count = df.duplicated().sum()
    info["duplicates"] = {
        "duplicate_rows": int(duplicate_count),
        "duplicate_percentage": round(duplicate_count / len(df) * 100, 2) if len(df) > 0 else 0
    }
    
    # Column cardinality (uniqueness)
    cardinality = {}
    for col in df.columns:
        unique_count = df[col].nunique()
        cardinality[col] = {
            "unique_count": int(unique_count),
            "unique_percentage": round(unique_count / len(df) * 100, 2) if len(df) > 0 else 0
        }
    
    info["cardinality"] = cardinality
    
    # Value distributions for categorical columns (limited to avoid huge outputs)
    distributions = {}
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_columns:
        if df[col].nunique() <= 10:  # Only for columns with reasonable cardinality
            value_counts = df[col].value_counts(normalize=True).head(10)
            distributions[col] = {str(k): float(v) for k, v in value_counts.items()}
    
    info["distributions"] = distributions
    
    return info


def detect_anomalies(df: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5) -> Dict[str, List[int]]:
    """
    Detect anomalies in numeric columns of a dataframe
    
    Args:
        df: Input dataframe
        method: Method for anomaly detection ('iqr', 'zscore', or 'isolation_forest')
        threshold: Threshold for outlier detection
        
    Returns:
        Dictionary with column names as keys and lists of anomalous row indices as values
    """
    anomalies = {}
    
    # Only process numeric columns
    numeric_columns = df.select_dtypes(include=['number']).columns
    
    if method == 'iqr':
        # Interquartile Range method
        for col in numeric_columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            # Get indices of outliers
            outlier_indices = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
            
            if outlier_indices:
                anomalies[col] = outlier_indices
    
    elif method == 'zscore':
        # Z-score method
        from scipy import stats
        
        for col in numeric_columns:
            z_scores = stats.zscore(df[col], nan_policy='omit')
            # Use absolute z-score
            outlier_indices = df[abs(z_scores) > threshold].index.tolist()
            
            if outlier_indices:
                anomalies[col] = outlier_indices
    
    elif method == 'isolation_forest':
        # Isolation Forest method (more complex but more powerful)
        try:
            from sklearn.ensemble import IsolationForest
            
            # Only use columns with enough values
            valid_columns = [col for col in numeric_columns 
                           if df[col].notna().sum() > len(df) * 0.5]
            
            if valid_columns:
                # Prepare data
                X = df[valid_columns].fillna(df[valid_columns].mean())
                
                # Train isolation forest
                model = IsolationForest(contamination=0.1, random_state=42)
                model.fit(X)
                
                # Predict anomalies
                anomaly_flags = model.predict(X)
                outlier_indices = df[anomaly_flags == -1].index.tolist()
                
                if outlier_indices:
                    anomalies['multivariate_outliers'] = outlier_indices
        except Exception as e:
            logger.warning(f"Isolation Forest anomaly detection failed: {e}")
    
    return anomalies


def batch_process_files(file_paths: List[Path], 
                       output_dir: Path, 
                       status_dir: Path, 
                       settings: Dict = None) -> Dict:
    """
    Process multiple files with common settings
    
    Args:
        file_paths: List of paths to input files
        output_dir: Directory to save output files
        status_dir: Directory to save status files
        settings: Optional dictionary of processing settings
        
    Returns:
        Dictionary with processing results for each file
    """
    results = {}
    
    # Process each file
    for file_path in file_paths:
        try:
            logger.info(f"Processing file: {file_path}")
            result = process_data_file(file_path, output_dir, status_dir)
            results[str(file_path)] = result
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            results[str(file_path)] = {
                "success": False,
                "error": str(e)
            }
    
    # Summarize the batch processing
    summary = {
        "total_files": len(file_paths),
        "successful": sum(1 for r in results.values() if r.get("success", False)),
        "failed": sum(1 for r in results.values() if not r.get("success", False)),
        "results": results
    }
    
    # Save summary to a file
    summary_path = output_dir / f"batch_summary_{int(time.time())}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python data_preprocessing.py input_file.csv [output_dir] [status_dir]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./processed_files"
    status_dir = sys.argv[3] if len(sys.argv) > 3 else "./status_files"
    
    # Simple progress callback
    def print_progress(progress, message):
        print(f"Progress: {progress}% - {message}")
    
    # Process the file
    result = process_data_file(input_file, output_dir, status_dir, print_progress)
    
    # Print the result
    print("\nProcessing Result:")
    print(json.dumps(result, indent=2))