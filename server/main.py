from fastapi import FastAPI, File, UploadFile, HTTPException
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:3000",  # Adjust this to your Next.js app's URL
    "https://your-next-app-domain.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a folder to store uploaded files if it doesn't exist
upload_folder = Path("files")
upload_folder.mkdir(exist_ok=True)

@app.post("/upload/")
async def upload_files(files: list[UploadFile] = File(...)):
    saved_files = []

    for file in files:
        # Ensure the file is either CSV or XLSX
        if file.content_type not in ["text/csv", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
            raise HTTPException(status_code=400, detail=f"File type not supported for {file.filename}. Only CSV or XLSX allowed.")
        
        # Save the file to the local folder
        file_path = upload_folder / file.filename
        with file_path.open("wb") as f:
            content = await file.read()
            f.write(content)
        
        saved_files.append(file.filename)
    
    return {"message": f"{len(saved_files)} file(s) uploaded successfully.", "files": saved_files}
