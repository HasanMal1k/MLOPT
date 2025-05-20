from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import logging
import os
import time
import json
import uuid
import tempfile
from typing import List

# Import custom modules
from data_preprocessing import process_data_file, update_status, generate_eda_report
from custom_preprocessing import router as custom_preprocessing_router
from transformations import router as transformations_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mlopt_server')

# Initialize FastAPI app
app = FastAPI(
    title="MLOpt Data Preprocessing API",
    description="API for data preprocessing and feature engineering for machine learning",
    version="1.0.0"
)

# Register custom routers
app.include_router(custom_preprocessing_router)
app.include_router(transformations_router)

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
    Upload and preprocess multiple files
    
    This endpoint accepts CSV and XLSX files, saves them,
    and triggers background preprocessing tasks for each file.
    """
    saved_files = []
    processing_info = []
    
    for file in files:
        # Check file type
        file_extension = file.filename.lower().split('.')[-1]
        is_valid_extension = file_extension in ['csv', 'xlsx', 'xls']
        
        if not is_valid_extension:
            raise HTTPException(
                status_code=400, 
                detail=f"File type not supported for {file.filename}. Only CSV, XLS, or XLSX allowed."
            )
        
        # Generate a unique filename to prevent collisions
        unique_id = str(uuid.uuid4())[:8]
        safe_filename = f"{unique_id}_{file.filename}"
        
        # Save the file to the local folder
        file_path = UPLOAD_FOLDER / safe_filename
        with file_path.open("wb") as f:
            content = await file.read()
            f.write(content)
        
        saved_files.append({
            "original_filename": file.filename,
            "saved_filename": safe_filename
        })
        
        # Process the file in the background
        output_filename = f"processed_{safe_filename}"
        output_path = PROCESSED_FOLDER / output_filename
        
        # Initialize progress tracking
        init_status = update_status(safe_filename, STATUS_FOLDER, 0, "Queued for processing")
        processing_info.append({
            "filename": safe_filename,
            "original_filename": file.filename,
            "processed_filename": output_filename,
            "status": init_status
        })
        
        # Add the background task for processing
        background_tasks.add_task(
            process_data_file,
            file_path,
            PROCESSED_FOLDER,
            STATUS_FOLDER,
            lambda progress, message: update_status(safe_filename, STATUS_FOLDER, progress, message)
        )
    
    return JSONResponse(
        content={
            "message": f"{len(saved_files)} file(s) uploaded successfully.",
            "files": saved_files,
            "processing": "File preprocessing started in the background. Track progress using the /processing-status/{filename} endpoint.",
            "processing_info": processing_info
        }
    )

@app.get("/processed-files/")
async def get_processed_files():
    """Get a list of all processed files"""
    files = [f.name for f in PROCESSED_FOLDER.glob("*") if f.is_file() and not (f.name.endswith('.joblib') or f.name.endswith('.json'))]
    return {"processed_files": files}

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Get a processed file by filename"""
    file_path = PROCESSED_FOLDER / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return {"file_url": f"/processed_files/{filename}"}

@app.get("/processing-status/{filename}")
async def get_processing_status(filename: str):
    """Get the processing status and progress of a file"""
    status_path = STATUS_FOLDER / f"{filename}_status.json"
    
    if status_path.exists():
        with open(status_path, 'r') as f:
            status = json.load(f)
            
        # If processing is complete, add preprocessing details
        if status.get("progress") == 100:
            pipeline_path = PROCESSED_FOLDER / f"processed_{filename}_pipeline.joblib"
            if pipeline_path.exists():
                try:
                    preprocessing_data = joblib.load(pipeline_path)
                    status["preprocessing_details"] = preprocessing_data["preprocessing_info"]
                except Exception as e:
                    logger.error(f"Error loading pipeline data: {e}")
        
        return status
    else:
        return {"status": "not_found", "message": f"No processing status found for {filename}"}
    
@app.post("/generate_report/", response_class=HTMLResponse)
async def generate_report(file: UploadFile = File(...)):
    """
    Generate an Exploratory Data Analysis (EDA) report for a file
    
    This endpoint accepts a CSV or XLSX file and returns an HTML report
    with data profiling information.
    """
    # Check file extension
    file_extension = file.filename.lower().split('.')[-1]
    if file_extension not in ['csv', 'xlsx', 'xls']:
        raise HTTPException(status_code=400, detail="Only CSV or Excel files are accepted")

    try:
        # Read the file content
        contents = await file.read()
        
        # Log file details for debugging
        logger.info(f"Received file: {file.filename}, size: {len(contents)} bytes")
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp:
            tmp.write(contents)
            temp_path = tmp.name
        
        try:
            # Read the file with pandas based on file extension
            import pandas as pd
            if file_extension == 'csv':
                df = pd.read_csv(temp_path)
            else:  # Excel file
                df = pd.read_excel(temp_path)
            
            # Limit the rows to prevent very large reports
            if len(df) > 10000:
                df = df.sample(10000, random_state=42)
                
            # Generate the report
            html_report = generate_eda_report(df)
            
        except Exception as e:
            # If there's an error in processing, return a simple error report
            logger.error(f"Error processing file for EDA report: {e}")
            html_report = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Error Processing File</title>
                <style>
                    body {{ font-family: Arial, sans-serif; padding: 20px; }}
                    .error {{ color: red; background-color: #ffeeee; padding: 10px; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <h1>File Processing Error</h1>
                <p>There was an error processing the file: {file.filename}</p>
                <div class="error">
                    <h3>Error Details:</h3>
                    <pre>{str(e)}</pre>
                </div>
            </body>
            </html>
            """
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

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
    
    return {
        "batch_id": batch_id,
        "message": f"Batch processing started for {len(results)} files",
        "results": results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)