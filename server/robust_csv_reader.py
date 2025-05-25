import pandas as pd
import chardet
import io
import csv
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

def detect_csv_delimiter(text_content: str, sample_lines: int = 10) -> str:
    """
    Detect the delimiter used in a CSV file
    """
    # Get a sample of the text
    lines = text_content.split('\n')[:sample_lines]
    sample_text = '\n'.join(lines)
    
    # Try to detect delimiter using csv.Sniffer
    try:
        sniffer = csv.Sniffer()
        delimiter = sniffer.sniff(sample_text, delimiters=',;\t|').delimiter
        logger.info(f"Detected delimiter: '{delimiter}'")
        return delimiter
    except:
        # Fallback: count occurrences of common delimiters
        delimiters = [',', ';', '\t', '|']
        delimiter_counts = {}
        
        for delimiter in delimiters:
            count = sample_text.count(delimiter)
            if count > 0:
                delimiter_counts[delimiter] = count
        
        if delimiter_counts:
            best_delimiter = max(delimiter_counts.items(), key=lambda x: x[1])[0]
            logger.info(f"Detected delimiter by counting: '{best_delimiter}'")
            return best_delimiter
        
        # Default to comma
        logger.warning("Could not detect delimiter, using comma")
        return ','

def detect_encoding_from_bytes(file_content: bytes) -> str:
    """
    Detect encoding from file content bytes with improved fallback
    """
    try:
        result = chardet.detect(file_content)
        encoding = result['encoding']
        confidence = result['confidence']
        
        logger.info(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
        
        # If confidence is very low, try some heuristics
        if confidence < 0.5:
            # Check for common Windows encoding indicators
            if b'\xfc' in file_content or b'\xe4' in file_content or b'\xf6' in file_content:
                logger.info("Detected German/European characters, trying cp1252")
                return 'cp1252'
            elif b'\xff\xfe' in file_content[:2] or b'\xfe\xff' in file_content[:2]:
                logger.info("Detected BOM, trying utf-16")
                return 'utf-16'
        
        # If confidence is reasonable, use detected encoding
        if confidence >= 0.5 and encoding:
            return encoding
        
        # Default fallback
        return 'utf-8'
        
    except Exception as e:
        logger.warning(f"Encoding detection failed: {e}, using utf-8")
        return 'utf-8'

def read_csv_with_robust_handling(file_content: bytes) -> pd.DataFrame:
    """
    Read CSV with robust error handling for malformed files
    """
    # Step 1: Detect and decode encoding
    detected_encoding = detect_encoding_from_bytes(file_content)
    
    encodings_to_try = [
        detected_encoding,
        'utf-8',
        'utf-8-sig',
        'cp1252',  # Windows Western European
        'latin1',  # Very permissive
        'ascii',
    ]
    
    # Remove duplicates while preserving order
    encodings_to_try = list(dict.fromkeys([enc for enc in encodings_to_try if enc]))
    
    text_content = None
    used_encoding = None
    
    # Try each encoding until one works
    for encoding in encodings_to_try:
        try:
            text_content = file_content.decode(encoding)
            used_encoding = encoding
            logger.info(f"Successfully decoded with {encoding}")
            break
        except UnicodeDecodeError:
            logger.debug(f"Encoding {encoding} failed")
            continue
    
    # Final fallback with error replacement
    if text_content is None:
        try:
            text_content = file_content.decode('utf-8', errors='replace')
            used_encoding = 'utf-8-with-replacement'
            logger.warning("Using UTF-8 with error replacement")
        except Exception as e:
            raise ValueError(f"Cannot decode file content: {e}")
    
    # Step 2: Detect delimiter
    delimiter = detect_csv_delimiter(text_content)
    
    # Step 3: Try to read CSV with increasingly permissive settings
    string_io = io.StringIO(text_content)
    
    # Reading strategies in order of preference
    reading_strategies = [
        # Strategy 1: Standard reading
        {
            'sep': delimiter,
            'on_bad_lines': 'warn',
            'quoting': csv.QUOTE_MINIMAL,
            'skipinitialspace': True
        },
        # Strategy 2: More permissive
        {
            'sep': delimiter,
            'on_bad_lines': 'skip',
            'quoting': csv.QUOTE_ALL,
            'skipinitialspace': True,
            'skip_blank_lines': True
        },
        # Strategy 3: Very permissive
        {
            'sep': delimiter,
            'on_bad_lines': 'skip',
            'quoting': csv.QUOTE_NONE,
            'skipinitialspace': True,
            'skip_blank_lines': True,
            'engine': 'python'  # Python engine is more tolerant
        },
        # Strategy 4: Fallback with comma delimiter
        {
            'sep': ',',
            'on_bad_lines': 'skip',
            'quoting': csv.QUOTE_MINIMAL,
            'skipinitialspace': True,
            'skip_blank_lines': True,
            'engine': 'python'
        },
        # Strategy 5: Last resort - read as single column and try to split
        {
            'sep': None,  # This will be handled specially
            'on_bad_lines': 'skip',
            'engine': 'python'
        }
    ]
    
    df = None
    strategy_used = None
    
    for i, strategy in enumerate(reading_strategies):
        try:
            # Reset string IO position
            string_io.seek(0)
            
            if strategy.get('sep') is None:
                # Special case: read as single column and try to split later
                df = pd.read_csv(string_io, header=None, names=['raw_data'])
                
                # Try to split the first few rows to detect structure
                if len(df) > 0:
                    sample_row = str(df.iloc[0]['raw_data'])
                    for test_delimiter in [',', ';', '\t', '|']:
                        if test_delimiter in sample_row:
                            # Re-read with detected delimiter
                            string_io.seek(0)
                            df = pd.read_csv(string_io, sep=test_delimiter, on_bad_lines='skip', engine='python')
                            break
            else:
                df = pd.read_csv(string_io, **strategy)
            
            # Validate the result
            if df is not None and len(df) > 0 and len(df.columns) > 0:
                strategy_used = i + 1
                logger.info(f"Successfully read CSV using strategy {strategy_used}")
                break
            
        except Exception as e:
            logger.debug(f"Strategy {i + 1} failed: {e}")
            continue
    
    if df is None or len(df) == 0:
        raise ValueError("Could not read CSV file with any strategy")
    
    # Post-processing: clean up the dataframe
    try:
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Remove columns that are completely empty
        df = df.dropna(axis=1, how='all')
        
        # Clean column names (remove extra whitespace)
        df.columns = df.columns.astype(str).str.strip()
        
        # Handle duplicate column names
        if len(df.columns) != len(df.columns.unique()):
            df.columns = [f"{col}_{i}" if list(df.columns).count(col) > 1 else col 
                         for i, col in enumerate(df.columns)]
        
        logger.info(f"Final CSV shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        
    except Exception as e:
        logger.warning(f"Error in post-processing: {e}")
    
    return df

# Update the main reading function
def read_csv_with_encoding_detection(file_content: bytes) -> pd.DataFrame:
    """
    Enhanced CSV reader with robust error handling
    """
    return read_csv_with_robust_handling(file_content)

# Alternative function for specific problematic files
def read_problematic_csv(file_content: bytes, 
                        expected_columns: Optional[int] = None,
                        force_delimiter: Optional[str] = None) -> pd.DataFrame:
    """
    Special function for handling very problematic CSV files
    
    Args:
        file_content: Raw file bytes
        expected_columns: If you know how many columns there should be
        force_delimiter: If you want to force a specific delimiter
    """
    # Decode content
    try:
        # Try common encodings
        for encoding in ['utf-8', 'cp1252', 'latin1']:
            try:
                text_content = file_content.decode(encoding)
                logger.info(f"Decoded with {encoding}")
                break
            except:
                continue
        else:
            text_content = file_content.decode('utf-8', errors='replace')
            logger.warning("Used UTF-8 with replacement")
    except Exception as e:
        raise ValueError(f"Cannot decode file: {e}")
    
    # If force_delimiter is specified, use it
    if force_delimiter:
        delimiter = force_delimiter
    else:
        delimiter = detect_csv_delimiter(text_content)
    
    # Try to read line by line and fix issues
    lines = text_content.split('\n')
    cleaned_lines = []
    
    # Analyze the structure
    header_line = lines[0] if lines else ""
    expected_fields = len(header_line.split(delimiter)) if not expected_columns else expected_columns
    
    logger.info(f"Expected fields per line: {expected_fields}")
    
    for i, line in enumerate(lines):
        if not line.strip():
            continue
            
        fields = line.split(delimiter)
        
        if len(fields) == expected_fields:
            cleaned_lines.append(line)
        elif len(fields) > expected_fields:
            # Too many fields - keep only the first expected_fields
            cleaned_line = delimiter.join(fields[:expected_fields])
            cleaned_lines.append(cleaned_line)
            logger.debug(f"Line {i+1}: Truncated from {len(fields)} to {expected_fields} fields")
        else:
            # Too few fields - pad with empty strings
            missing_fields = expected_fields - len(fields)
            padded_fields = fields + [''] * missing_fields
            cleaned_line = delimiter.join(padded_fields)
            cleaned_lines.append(cleaned_line)
            logger.debug(f"Line {i+1}: Padded from {len(fields)} to {expected_fields} fields")
    
    # Reconstruct the CSV content
    cleaned_content = '\n'.join(cleaned_lines)
    
    # Read the cleaned content
    try:
        string_io = io.StringIO(cleaned_content)
        df = pd.read_csv(string_io, sep=delimiter)
        logger.info(f"Successfully read cleaned CSV: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to read even after cleaning: {e}")
        raise ValueError(f"Cannot read CSV even after cleaning: {e}")