import pandas as pd
import chardet
from pathlib import Path
from typing import Tuple, Optional, List
import logging
import io
import tempfile
import os

logger = logging.getLogger(__name__)

def auto_convert_to_utf8(file_content: bytes) -> Tuple[str, str]:
    """
    Automatically detect encoding and convert file content to UTF-8 string
    """
    try:
        result = chardet.detect(file_content)
        detected_encoding = result['encoding']
        confidence = result['confidence']
        
        logger.info(f"Detected encoding: {detected_encoding} (confidence: {confidence:.2f})")
        
    except Exception as e:
        logger.warning(f"Encoding detection failed: {e}, defaulting to utf-8")
        detected_encoding = 'utf-8'
    
    # Common encodings to try in order of preference
    encodings_to_try = [
        detected_encoding,
        'utf-8',
        'utf-8-sig',  # UTF-8 with BOM
        'latin1',     # ISO-8859-1 (very permissive)
        'cp1252',     # Windows-1252
        'ascii',
    ]
    
    # Remove None and duplicates while preserving order
    encodings_to_try = list(dict.fromkeys([enc for enc in encodings_to_try if enc]))
    
    # Try each encoding until one works
    for encoding in encodings_to_try:
        try:
            text_content = file_content.decode(encoding)
            logger.info(f"Successfully decoded with {encoding}")
            return text_content, encoding
            
        except UnicodeDecodeError:
            logger.debug(f"Failed to decode with {encoding}")
            continue
        except Exception as e:
            logger.debug(f"Error with {encoding}: {e}")
            continue
    
    # Final fallback: decode with error replacement
    try:
        text_content = file_content.decode('utf-8', errors='replace')
        logger.warning("Used UTF-8 with error replacement - some characters may appear as �")
        return text_content, 'utf-8-with-replacement'
    except Exception as e:
        logger.error(f"Even fallback decoding failed: {e}")
        raise ValueError("Cannot decode file content to text")

def read_any_file_universal(file_content: bytes, filename: str, missing_values: List[str] = None) -> Tuple[pd.DataFrame, bool, str]:
    """
    Universal file reader that handles CSV, XLSX, and XLS with automatic encoding detection
    
    Args:
        file_content: Raw file content as bytes
        filename: Original filename (used to determine file type)
        missing_values: List of strings to interpret as missing values
        
    Returns:
        Tuple of (dataframe, success, message)
    """
    file_extension = filename.lower().split('.')[-1]
    
    try:
        if file_extension == 'csv':
            # Handle CSV with encoding detection
            utf8_content, detected_encoding = auto_convert_to_utf8(file_content)
            string_io = io.StringIO(utf8_content)
            
            df = pd.read_csv(
                string_io,
                na_values=missing_values,
                keep_default_na=True,
                on_bad_lines='warn'
            )
            
            success_msg = f"Successfully read CSV (converted from {detected_encoding} to UTF-8)"
            
        elif file_extension in ['xlsx', 'xls']:
            # Handle Excel files
            # Create a temporary file since pandas.read_excel needs a file path or file-like object
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name
            
            try:
                df = pd.read_excel(
                    tmp_file_path,
                    na_values=missing_values,
                    keep_default_na=True
                )
                success_msg = f"Successfully read {file_extension.upper()} file"
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
        
        else:
            return pd.DataFrame(), False, f"Unsupported file format: {file_extension}"
        
        # Validate the data
        if len(df) == 0:
            return df, False, f"File appears to be empty (0 rows)"
        
        if len(df.columns) == 0:
            return df, False, f"No columns detected in file"
        
        logger.info(f"Successfully read {filename}: {len(df)} rows, {len(df.columns)} columns")
        return df, True, success_msg
        
    except Exception as e:
        error_msg = f"Failed to read {filename}: {str(e)}"
        logger.error(error_msg)
        return pd.DataFrame(), False, error_msg

def read_any_file_from_path(file_path: Path, missing_values: List[str] = None) -> Tuple[pd.DataFrame, bool, str]:
    """
    Universal file reader for files on disk
    
    Args:
        file_path: Path to the file
        missing_values: List of strings to interpret as missing values
        
    Returns:
        Tuple of (dataframe, success, message)
    """
    try:
        # Read file as bytes
        with open(file_path, 'rb') as f:
            file_content = f.read()
        
        # Use the universal bytes reader
        return read_any_file_universal(file_content, file_path.name, missing_values)
        
    except Exception as e:
        error_msg = f"Failed to read file {file_path}: {str(e)}"
        logger.error(error_msg)
        return pd.DataFrame(), False, error_msg

def normalize_any_file(file_content: bytes, filename: str, output_dir: Path) -> Tuple[Path, str]:
    """
    Normalize any file (CSV gets UTF-8 conversion, Excel files saved as-is)
    
    Args:
        file_content: Raw file content as bytes
        filename: Original filename
        output_dir: Directory to save the normalized file
        
    Returns:
        Tuple of (path_to_normalized_file, conversion_message)
    """
    file_extension = filename.lower().split('.')[-1]
    normalized_filename = f"normalized_{filename}"
    output_path = output_dir / normalized_filename
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if file_extension == 'csv':
            # Convert CSV to UTF-8
            utf8_content, detected_encoding = auto_convert_to_utf8(file_content)
            
            # Save as UTF-8
            with open(output_path, 'w', encoding='utf-8', newline='') as f:
                f.write(utf8_content)
            
            message = f"CSV converted from {detected_encoding} to UTF-8"
            logger.info(f"Normalized CSV file: {output_path}")
            
        elif file_extension in ['xlsx', 'xls']:
            # Excel files don't need encoding conversion, save as-is
            with open(output_path, 'wb') as f:
                f.write(file_content)
            
            message = f"{file_extension.upper()} file saved (no encoding conversion needed)"
            logger.info(f"Saved Excel file: {output_path}")
            
        else:
            # Unknown format, save as-is
            with open(output_path, 'wb') as f:
                f.write(file_content)
            
            message = f"Unknown format ({file_extension}), saved as-is"
            logger.warning(f"Unknown format saved: {output_path}")
        
        return output_path, message
        
    except Exception as e:
        # Fallback: save as-is
        try:
            with open(output_path, 'wb') as f:
                f.write(file_content)
            error_message = f"Error during normalization ({str(e)}), saved as-is"
            logger.error(f"Normalization failed for {filename}: {e}")
            return output_path, error_message
        except Exception as save_error:
            logger.error(f"Failed to save file {filename}: {save_error}")
            raise Exception(f"Cannot save file: {save_error}")

def universal_safe_read_file(file_path: Path, missing_values: List[str] = None) -> Tuple[pd.DataFrame, bool, str]:
    """
    Universal replacement for your existing safe_read_file function
    Handles CSV (with encoding detection) and Excel files
    
    This is the main function to replace your existing safe_read_file
    """
    return read_any_file_from_path(file_path, missing_values)

def validate_file_before_processing(file_content: bytes, filename: str, missing_values: List[str] = None) -> Tuple[bool, str, dict]:
    """
    Validate that a file can be read before processing
    Returns validation status, message, and file info
    
    Args:
        file_content: Raw file content as bytes
        filename: Original filename
        missing_values: List of strings to interpret as missing values
        
    Returns:
        Tuple of (is_valid, message, file_info_dict)
    """
    try:
        df, success, message = read_any_file_universal(file_content, filename, missing_values)
        
        if not success:
            return False, message, {}
        
        file_info = {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "file_size_bytes": len(file_content),
            "file_type": filename.lower().split('.')[-1]
        }
        
        return True, message, file_info
        
    except Exception as e:
        return False, f"Validation error: {str(e)}", {}

# Compatibility functions for your existing code
def safe_read_file(file_path: Path, missing_values: List[str] = None) -> Tuple[pd.DataFrame, bool, str]:
    """
    Drop-in replacement for your existing safe_read_file function
    """
    return universal_safe_read_file(file_path, missing_values)

# Enhanced version that tries Polars first (if you want to keep Polars)
def enhanced_safe_read_file_with_polars(file_path: Path, missing_values: List[str] = None) -> Tuple[pd.DataFrame, bool, str]:
    """
    Enhanced version that tries Polars first for CSV files, then falls back to universal reader
    """
    file_extension = file_path.suffix.lower()
    
    # Only try Polars for CSV files
    if file_extension == '.csv':
        try:
            import polars as pl
            logger.info(f"Trying Polars for {file_path}")
            df = pl.read_csv(file_path, null_values=missing_values or [], ignore_errors=True)
            logger.info("Polars read successful")
            return df.to_pandas(), True, "Success with Polars"
        except Exception as e:
            logger.warning(f"Polars failed for {file_path}: {e}, falling back to universal reader")
    
    # Fall back to universal reader for all files
    return universal_safe_read_file(file_path, missing_values)