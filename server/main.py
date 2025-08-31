from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import logging
import os
import time
import json
import uuid
import tempfile
import math
import pandas as pd
import numpy as np
import chardet
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
# Import your modules
from data_preprocessing import process_data_file, update_status, generate_eda_report, MISSING_VALUES
from custom_preprocessing import router as custom_preprocessing_router
from transformations import router as transformations_router
from time_series_preprocessing import router as time_series_router
from universal_file_handler import (
    normalize_any_file, 
    validate_file_before_processing,
    read_any_file_universal
)
from robust_csv_reader import read_csv_with_robust_handling, read_problematic_csv
from data_visualization import router as visualization_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mlopt_server')

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

def detect_encoding_from_bytes(file_content: bytes) -> str:
    """
    Detect encoding from file content bytes
    """
    try:
        result = chardet.detect(file_content)
        encoding = result['encoding']
        confidence = result['confidence']
        
        logger.info(f"EDA: Detected encoding: {encoding} (confidence: {confidence:.2f})")
        
        # If confidence is low, default to utf-8
        if confidence < 0.7:
            logger.warning(f"Low confidence ({confidence:.2f}), defaulting to utf-8")
            return 'utf-8'
        
        return encoding
        
    except Exception as e:
        logger.warning(f"Encoding detection failed: {e}, using utf-8")
        return 'utf-8'

# Import the robust CSV reader at the top of your file
from robust_csv_reader import read_csv_with_robust_handling, read_problematic_csv

def read_csv_with_encoding_detection(file_content: bytes) -> pd.DataFrame:
    """
    Read CSV with robust error handling for malformed files
    """
    try:
        # First try the robust handler
        return read_csv_with_robust_handling(file_content)
    except Exception as e:
        logger.warning(f"Robust handler failed: {e}, trying problematic CSV handler")
        
        # If that fails, try the special problematic CSV handler
        try:
            return read_problematic_csv(file_content)
        except Exception as e2:
            logger.error(f"All CSV reading methods failed: {e2}")
            raise ValueError(f"Cannot read CSV file: {e2}")


# Initialize FastAPI app
app = FastAPI(
    title="MLOpt Data Preprocessing API",
    description="API for data preprocessing and feature engineering for machine learning",
    version="1.0.0"
)

# Register custom routers
app.include_router(custom_preprocessing_router)
app.include_router(transformations_router)
app.include_router(time_series_router)  # Register the time series router
app.include_router(visualization_router)


# Configure CORS
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "https://your-production-domain.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create folders to store uploaded and processed files
UPLOAD_FOLDER = Path("files")
PROCESSED_FOLDER = Path("processed_files")
STATUS_FOLDER = Path("processing_status")

# Create directories if they don't exist
UPLOAD_FOLDER.mkdir(exist_ok=True)
PROCESSED_FOLDER.mkdir(exist_ok=True)
STATUS_FOLDER.mkdir(exist_ok=True)

@app.get("/")
async def root():
    """Root endpoint for API health check"""
    return {"status": "online", "message": "MLOpt API is running"}

@app.post("/upload/")
async def upload_files(background_tasks: BackgroundTasks, files: list[UploadFile] = File(...)):
    """
    Universal upload endpoint that handles CSV (with encoding detection) and Excel files
    """
    try:
        saved_files = []
        processing_info = []
        
        for file in files:
            # Check file type
            if not file.filename:
                continue
                
            file_extension = file.filename.lower().split('.')[-1]
            is_valid_extension = file_extension in ['csv', 'xlsx', 'xls']
            
            if not is_valid_extension:
                logger.warning(f"Invalid file type for {file.filename}: {file_extension}")
                continue
            
            # Generate a unique filename to prevent collisions
            unique_id = str(uuid.uuid4())[:8]
            safe_filename = f"{unique_id}_{file.filename}"
            
            try:
                # Read file content
                file_content = await file.read()
                
                # Validate the file can be read before processing
                is_valid, validation_message, file_info = validate_file_before_processing(
                    file_content, file.filename, MISSING_VALUES
                )
                
                if not is_valid:
                    logger.error(f"File validation failed for {file.filename}: {validation_message}")
                    processing_info.append({
                        "filename": safe_filename,
                        "original_filename": file.filename,
                        "status": "validation_error",
                        "error": f"File validation failed: {validation_message}"
                    })
                    continue
                
                logger.info(f"File validation passed for {file.filename}: {validation_message}")
                logger.info(f"File info: {file_info}")
                
                # Normalize the file (convert CSV to UTF-8, save Excel as-is)
                normalized_path, conversion_message = normalize_any_file(
                    file_content, safe_filename, UPLOAD_FOLDER
                )
                
                logger.info(f"File normalization: {conversion_message}")
                
                saved_files.append({
                    "original_filename": file.filename,
                    "saved_filename": normalized_path.name,
                    "file_type": file_extension,
                    "conversion_message": conversion_message,
                    "file_info": file_info
                })
                
                # The processed file will be saved as "processed_{safe_filename}"
                processed_filename = f"processed_{safe_filename}"
                
                # Initialize progress tracking
                try:
                    initial_status_data = {
                        "original_filename": file.filename,
                        "safe_filename": normalized_path.name,
                        "processed_filename": processed_filename,
                        "file_type": file_extension,
                        "conversion_message": conversion_message,
                        "file_info": file_info,
                        "file_mapping": {
                            "input": normalized_path.name,
                            "output": processed_filename
                        }
                    }
                    
                    init_status = update_status(normalized_path.name, STATUS_FOLDER, 0, "Queued for processing", initial_status_data)
                    processing_info.append({
                        "filename": normalized_path.name,
                        "original_filename": file.filename,
                        "processed_filename": processed_filename,
                        "file_type": file_extension,
                        "conversion_message": conversion_message,
                        "file_info": file_info,
                        "status": init_status
                    })
                    
                    # Add the background task for processing using normalized file
                    background_tasks.add_task(
                        process_data_file,
                        normalized_path,  # Use the normalized file path
                        PROCESSED_FOLDER,
                        STATUS_FOLDER
                    )
                    
                except Exception as status_error:
                    logger.error(f"Error setting up processing for {file.filename}: {status_error}")
                    continue
                    
            except Exception as file_error:
                logger.error(f"Error processing file {file.filename}: {file_error}")
                processing_info.append({
                    "filename": safe_filename,
                    "original_filename": file.filename,
                    "status": "processing_error",
                    "error": str(file_error)
                })
                continue
        
        if not saved_files:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "No valid files were uploaded",
                    "message": "Please upload CSV or Excel files. Check that files are not corrupted.",
                    "failed_files": [info for info in processing_info if "error" in info]
                }
            )
        
        response_data = {
            "message": f"{len(saved_files)} file(s) uploaded and processed successfully.",
            "files": saved_files,
            "processing": "Files have been automatically normalized (CSV files converted to UTF-8) and preprocessing started in the background.",
            "processing_info": processing_info,
            "summary": {
                "total_uploaded": len(saved_files),
                "csv_files": len([f for f in saved_files if f["file_type"] == "csv"]),
                "excel_files": len([f for f in saved_files if f["file_type"] in ["xlsx", "xls"]])
            }
        }
        
        return JSONResponse(content=clean_for_json(response_data))
        
    except Exception as e:
        logger.error(f"Error in upload endpoint: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Upload failed",
                "message": str(e)
            }
        )
        
# Optional: Add an endpoint to test file reading without uploading
@app.post("/test-file-reading/")
async def test_file_reading(file: UploadFile = File(...)):
    """
    Test endpoint to validate file reading without processing
    """
    try:
        file_content = await file.read()
        
        # Test reading the file
        is_valid, message, file_info = validate_file_before_processing(
            file_content, file.filename, MISSING_VALUES
        )
        
        if is_valid:
            # Also get a sample of the data
            df, success, read_message = read_any_file_universal(file_content, file.filename, MISSING_VALUES)
            
            if success and len(df) > 0:
                sample_data = df.head(5).to_dict('records')
                file_info['sample_data'] = sample_data
                file_info['columns_info'] = {
                    col: str(df[col].dtype) for col in df.columns
                }
        
        return JSONResponse(content={
            "filename": file.filename,
            "is_valid": is_valid,
            "message": message,
            "file_info": clean_for_json(file_info)
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Error testing file: {str(e)}"
            }
        )
        
@app.get("/processed-files/{filename}")
async def serve_processed_file(filename: str):
    """
    Serve processed files directly
    """
    try:
        # Try different possible file paths
        possible_paths = [
            PROCESSED_FOLDER / filename,
            PROCESSED_FOLDER / f"processed_{filename}",
        ]
        
        for file_path in possible_paths:
            if file_path.exists() and file_path.is_file():
                return FileResponse(
                    path=str(file_path),
                    filename=filename,
                    media_type='application/octet-stream'
                )
        
        # If no file found, return 404
        raise HTTPException(status_code=404, detail=f"File {filename} not found")
        
    except Exception as e:
        logger.error(f"Error serving file {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error serving file: {str(e)}")

@app.get("/download/{filename}")
async def download_processed_file(filename: str):
    """
    Download a processed file with proper headers
    """
    try:
        # Check both with and without 'processed_' prefix
        possible_paths = [
            PROCESSED_FOLDER / filename,
            PROCESSED_FOLDER / f"processed_{filename}",
        ]
        
        file_path = None
        for path in possible_paths:
            if path.exists() and path.is_file():
                file_path = path
                break
        
        if not file_path:
            raise HTTPException(status_code=404, detail=f"File {filename} not found")
        
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type='text/csv',
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Access-Control-Allow-Origin": "*"
            }
        )
        
    except Exception as e:
        logger.error(f"Error downloading file {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")


@app.get("/processing-status/{filename}")
async def get_processing_status(filename: str):
    """Get the processing status and progress of a file with file mapping info"""
    try:
        status_path = STATUS_FOLDER / f"{filename}_status.json"
        
        if not status_path.exists():
            logger.warning(f"Status file not found for {filename}")
            return clean_for_json({
                "status": "not_found", 
                "progress": 0,
                "message": f"No processing status found for {filename}",
                "results": None,
                "file_mapping": None
            })
        
        try:
            with open(status_path, 'r') as f:
                content = f.read().strip()
                
                if not content:
                    logger.warning(f"Empty status file for {filename}")
                    return clean_for_json({
                        "status": "processing", 
                        "progress": 0,
                        "message": "Status file is empty - processing may have just started",
                        "results": None,
                        "file_mapping": None
                    })
                
                status = json.loads(content)
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for {filename}: {e}")
            return clean_for_json({
                "status": "error", 
                "progress": -1,
                "message": f"Error reading status file: JSON decode error",
                "results": None,
                "file_mapping": None
            })
        except Exception as e:
            logger.error(f"Error reading status file for {filename}: {e}")
            return clean_for_json({
                "status": "error", 
                "progress": -1,
                "message": f"Error reading status file: {str(e)}",
                "results": None,
                "file_mapping": None
            })
        
        # Ensure status has required fields
        status.setdefault("status", "processing")
        status.setdefault("progress", 0)
        status.setdefault("message", "Processing...")
        status.setdefault("results", None)
        status.setdefault("file_mapping", None)
        
        # If processing is complete, try to add preprocessing details
        if status.get("progress") == 100 and status.get("results") is None:
            pipeline_path = PROCESSED_FOLDER / f"processed_{filename}_pipeline.joblib"
            if pipeline_path.exists():
                try:
                    import joblib
                    preprocessing_data = joblib.load(pipeline_path)
                    if "preprocessing_info" in preprocessing_data:
                        status["results"] = preprocessing_data["preprocessing_info"]
                        logger.info(f"Added preprocessing results for {filename}")
                except Exception as e:
                    logger.error(f"Error loading pipeline data for {filename}: {e}")
        
        # Clean the status before returning
        return JSONResponse(content=clean_for_json(status))
        
    except Exception as e:
        logger.error(f"Unexpected error in get_processing_status for {filename}: {e}")
        return JSONResponse(content=clean_for_json({
            "status": "error", 
            "progress": -1,
            "message": f"Unexpected server error: {str(e)}",
            "results": None,
            "file_mapping": None
        }))
        
        
def update_status(filename: str, status_folder: Path, 
                 progress: int, message: str, 
                 results: Optional[Dict] = None) -> Dict:
    """
    Update the processing status of a file with better error handling and file tracking.
    """
    try:
        status = {
            "status": "processing" if progress < 100 and progress >= 0 else ("completed" if progress == 100 else "error"),
            "progress": progress,
            "message": message,
            "timestamp": time.time()
        }
        
        if results is not None:
            # Clean results before adding to status
            status["results"] = clean_for_json(results)
        
        # Add file mapping information
        if progress == 100:  # When processing is complete
            processed_filename = f"processed_{filename}"
            processed_path = PROCESSED_FOLDER / processed_filename
            
            status["file_mapping"] = {
                "original_filename": filename,
                "processed_filename": processed_filename,
                "processed_file_exists": processed_path.exists(),
                "processed_file_size": processed_path.stat().st_size if processed_path.exists() else 0
            }
            
            logger.info(f"File mapping for {filename}: processed file exists = {processed_path.exists()}")
        
        status_path = status_folder / f"{filename}_status.json"
        
        # Ensure the directory exists
        status_folder.mkdir(parents=True, exist_ok=True)
        
        # Clean the entire status object before writing
        clean_status = clean_for_json(status)
        
        # Write status with proper error handling
        with open(status_path, 'w') as f:
            json.dump(clean_status, f, indent=2)
        
        logger.info(f"Updated status for {filename}: {progress}% - {message}")
        return clean_status
        
    except Exception as e:
        logger.error(f"Error updating status for {filename}: {e}")
        # Return a basic status even if file write fails
        return clean_for_json({
            "status": "error",
            "progress": -1,
            "message": f"Error updating status: {str(e)}",
            "timestamp": time.time()
        })
        
@app.get("/list-processed-files/")
async def list_processed_files():
    """
    List all available processed files for debugging
    """
    try:
        files = []
        if PROCESSED_FOLDER.exists():
            for file_path in PROCESSED_FOLDER.iterdir():
                if file_path.is_file() and not file_path.name.endswith(('.joblib', '.json')):
                    files.append({
                        "filename": file_path.name,
                        "size": file_path.stat().st_size,
                        "exists": True
                    })
        
        response_data = {
            "processed_files": files,
            "count": len(files),
            "folder_path": str(PROCESSED_FOLDER)
        }
        
        return JSONResponse(content=clean_for_json(response_data))
        
    except Exception as e:
        logger.error(f"Error listing processed files: {e}")
        return JSONResponse(content=clean_for_json({
            "error": str(e), 
            "processed_files": [], 
            "count": 0
        }))

# Optional: Mount static files (add this after creating the app but before the endpoints)
# This creates a static file server for the processed_files directory
try:
    app.mount("/static/processed", StaticFiles(directory=str(PROCESSED_FOLDER)), name="processed_files")
    logger.info(f"Mounted static files from {PROCESSED_FOLDER}")
except Exception as e:
    logger.warning(f"Could not mount static files: {e}")
        
    
@app.post("/generate_report/", response_class=HTMLResponse)
async def generate_report(file: UploadFile = File(...)):
    """
    Generate an Exploratory Data Analysis (EDA) report for a file - now with encoding detection
    """
    # Check file extension
    file_extension = file.filename.lower().split('.')[-1]
    if file_extension not in ['csv', 'xlsx', 'xls']:
        raise HTTPException(status_code=400, detail="Only CSV or Excel files are accepted")

    temp_path = None
    try:
        # Read the file content
        contents = await file.read()
        
        # Log file details for debugging
        logger.info(f"Received file: {file.filename}, size: {len(contents)} bytes")
        
        # Handle different file types
        if file_extension == 'csv':
            # Use encoding detection for CSV files
            try:
                df = read_csv_with_encoding_detection(contents)
                logger.info(f"Successfully read CSV: {len(df)} rows, {len(df.columns)} columns")
                
            except Exception as e:
                logger.error(f"Failed to read CSV with encoding detection: {e}")
                raise Exception(f"Could not read CSV file: {e}")
        
        else:
            # Handle Excel files (create temporary file as before)
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp:
                tmp.write(contents)
                temp_path = tmp.name
            
            try:
                df = pd.read_excel(temp_path)
                logger.info(f"Successfully read Excel: {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                logger.error(f"Failed to read Excel file: {e}")
                raise Exception(f"Could not read Excel file: {e}")
        
        # Limit the rows to prevent very large reports
        original_rows = len(df)
        if len(df) > 10000:
            df = df.sample(10000, random_state=42)
            logger.info(f"Sampled {len(df)} rows from {original_rows} for EDA report")
            
        # Generate the report
        html_report = generate_eda_report(df)
        
        # Add info about sampling and encoding if applicable
        if original_rows > 10000:
            sampling_note = f"<p><strong>Note:</strong> This report is based on a random sample of 10,000 rows from the original {original_rows:,} rows.</p>"
            # Insert the note after the opening body tag
            html_report = html_report.replace('<body>', f'<body>{sampling_note}')
        
    except Exception as e:
        # If there's an error in processing, return a detailed error report
        logger.error(f"Error processing file for EDA report: {e}")
        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Error Processing File</title>
            <style>
                body {{ font-family: Arial, sans-serif; padding: 20px; }}
                .error {{ color: red; background-color: #ffeeee; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                .info {{ color: blue; background-color: #eeeeff; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                .suggestion {{ color: green; background-color: #eeffee; padding: 15px; border-radius: 5px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>File Processing Error</h1>
            <p>There was an error processing the file: <strong>{file.filename}</strong></p>
            
            <div class="error">
                <h3>Error Details:</h3>
                <pre>{str(e)}</pre>
            </div>
            
            <div class="info">
                <h3>File Information:</h3>
                <ul>
                    <li>Filename: {file.filename}</li>
                    <li>File size: {len(contents):,} bytes</li>
                    <li>File type: {file_extension.upper()}</li>
                </ul>
            </div>
            
            <div class="suggestion">
                <h3>Suggestions:</h3>
                <ul>
                    <li>If this is a CSV file, it may have encoding issues. Try saving it as UTF-8 from Excel or your text editor.</li>
                    <li>Check that the file is not corrupted.</li>
                    <li>For Excel files, try saving as a newer .xlsx format.</li>
                    <li>Make sure the file contains valid tabular data.</li>
                </ul>
            </div>
        </body>
        </html>
        """
    
    finally:
        # Clean up the temporary file if it was created
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass

    # Return the HTML with proper headers
    return HTMLResponse(
        content=html_report,
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Content-Type": "text/html"
        }
    )

@app.post("/batch-process/")
async def batch_process(
    background_tasks: BackgroundTasks, 
    files: List[UploadFile] = File(...),
    process_settings: str = Form("{}")
):
    """
    Batch process multiple files with custom settings
    
    This endpoint accepts multiple files and a JSON string of processing settings,
    and processes all files in the background using the specified settings.
    """
    # Parse process settings
    try:
        settings = json.loads(process_settings)
    except json.JSONDecodeError:
        settings = {}
    
    # Process each file
    batch_id = str(uuid.uuid4())[:8]
    results = []
    
    for file in files:
        # Check file type
        file_extension = file.filename.lower().split('.')[-1]
        if file_extension not in ['csv', 'xlsx', 'xls']:
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": f"Unsupported file format: {file_extension}"
            })
            continue
        
        # Save the file with a unique name
        safe_filename = f"{batch_id}_{file.filename}"
        file_path = UPLOAD_FOLDER / safe_filename
        
        with file_path.open("wb") as f:
            content = await file.read()
            f.write(content)
        
        # Add to background processing
        background_tasks.add_task(
            process_data_file,
            file_path,
            PROCESSED_FOLDER,
            STATUS_FOLDER,
            lambda progress, message: update_status(safe_filename, STATUS_FOLDER, progress, message)
        )
        
        results.append({
            "filename": file.filename,
            "saved_as": safe_filename,
            "status": "processing",
            "status_url": f"/processing-status/{safe_filename}"
        })
    
    response_data = {
        "batch_id": batch_id,
        "message": f"Batch processing started for {len(results)} files",
        "results": results
    }
    
    return JSONResponse(content=clean_for_json(response_data))

@app.get("/files/processed/{filename}")
async def get_processed_file_alt(filename: str):
    """
    Alternative endpoint to serve processed files with better path resolution
    """
    try:
        logger.info(f"Requesting processed file: {filename}")
        
        # List of possible file locations and names
        possible_files = [
            PROCESSED_FOLDER / filename,
            PROCESSED_FOLDER / f"processed_{filename}",
            PROCESSED_FOLDER / f"{filename.split('.')[0]}_processed.csv",
        ]
        
        # Also check for files that might have been renamed during processing
        if PROCESSED_FOLDER.exists():
            for existing_file in PROCESSED_FOLDER.iterdir():
                if existing_file.is_file() and filename.lower() in existing_file.name.lower():
                    possible_files.append(existing_file)
        
        # Try each possible file location
        for file_path in possible_files:
            if file_path.exists() and file_path.is_file():
                logger.info(f"Found processed file at: {file_path}")
                
                # Get file size for logging
                file_size = file_path.stat().st_size
                logger.info(f"Serving file: {file_path.name}, size: {file_size} bytes")
                
                return FileResponse(
                    path=str(file_path),
                    filename=file_path.name,
                    media_type='text/csv',
                    headers={
                        "Access-Control-Allow-Origin": "*",
                        "Content-Length": str(file_size)
                    }
                )
        
        # If no file found, log available files
        available_files = []
        if PROCESSED_FOLDER.exists():
            available_files = [f.name for f in PROCESSED_FOLDER.iterdir() if f.is_file()]
        
        logger.warning(f"File {filename} not found. Available files: {available_files}")
        
        raise HTTPException(
            status_code=404, 
            detail={
                "message": f"File {filename} not found",
                "available_files": available_files,
                "searched_paths": [str(p) for p in possible_files]
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving processed file {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error serving file: {str(e)}")

# Health check endpoint that also tests file system
@app.get("/health/files")
async def health_check_files():
    """
    Health check that includes file system status
    """
    try:
        health_info = {
            "status": "healthy",
            "timestamp": time.time(),
            "folders": {},
            "permissions": {}
        }
        
        # Check each required folder
        folders_to_check = {
            "upload": UPLOAD_FOLDER,
            "processed": PROCESSED_FOLDER,
            "status": STATUS_FOLDER
        }
        
        for folder_name, folder_path in folders_to_check.items():
            health_info["folders"][folder_name] = {
                "path": str(folder_path),
                "exists": folder_path.exists(),
                "is_directory": folder_path.is_dir() if folder_path.exists() else False,
                "file_count": len(list(folder_path.iterdir())) if folder_path.exists() and folder_path.is_dir() else 0
            }
            
            # Test write permissions
            try:
                test_file = folder_path / f"test_write_{int(time.time())}.tmp"
                folder_path.mkdir(exist_ok=True, parents=True)
                test_file.write_text("test")
                test_file.unlink()
                health_info["permissions"][folder_name] = "write_ok"
            except Exception as e:
                health_info["permissions"][folder_name] = f"write_error: {str(e)}"
        
        return JSONResponse(content=clean_for_json(health_info))
        
    except Exception as e:
        return JSONResponse(content=clean_for_json({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }))

try:
    from pathlib import Path
    transformed_folder = Path("transformed_files")
    transformed_folder.mkdir(exist_ok=True)
    app.mount("/static/transformed_files", StaticFiles(directory=str(transformed_folder)), name="transformed_files")
    logger.info(f"Mounted transformed files from {transformed_folder}")
except Exception as e:
    logger.warning(f"Could not mount transformed files: {e}")

# Also add this endpoint to serve transformed files directly
@app.get("/transformed-files/{filename}")
async def serve_transformed_file(filename: str):
    """
    Serve transformed files directly
    """
    try:
        from pathlib import Path
        transformed_folder = Path("transformed_files")
        file_path = transformed_folder / filename
        
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=404, detail=f"Transformed file {filename} not found")
        
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type='text/csv',
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Access-Control-Allow-Origin": "*"
            }
        )
        
    except Exception as e:
        logger.error(f"Error serving transformed file {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error serving file: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)