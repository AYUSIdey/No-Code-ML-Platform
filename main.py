from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

# Import our custom engines
from data_engine import analyze_dataset
from ml_engine import train_and_evaluate
from vision_engine import analyze_vision_dataset, train_vision_model

app = FastAPI(title="No-Code ML API")

# Configure CORS so our React frontend can talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for local development)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/api/upload-dataset/")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Receives a CSV file from the frontend, saves it temporarily, 
    and passes it to the data engine for EDA analysis.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported for Tabular data.")
        
    temp_file_path = f"temp_{file.filename}"
    
    try:
        # Save the uploaded file temporarily to disk
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Pass the file path to our Data Engine
        analysis_result = analyze_dataset(temp_file_path)
        
        if analysis_result.get("status") == "error":
            raise HTTPException(status_code=500, detail=analysis_result.get("message"))
            
        return analysis_result
        
    finally:
        # Clean up! Always delete the temporary file after we are done with it
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/api/upload-vision-dataset/")
async def upload_vision_dataset(file: UploadFile = File(...)):
    """
    Receives a ZIP file containing image folders, extracts it,
    and returns a breakdown of the classes for the EDA dashboard.
    """
    # Validate the file type
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Only ZIP files are supported for Computer Vision.")
    
    temp_file_path = f"temp_vision_{file.filename}"
    try:
        # Save the uploaded zip file temporarily to disk
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Pass the zip to our Vision Engine
        analysis_result = analyze_vision_dataset(temp_file_path)
        
        if analysis_result.get("status") == "error":
            raise HTTPException(status_code=500, detail=analysis_result.get("message"))
            
        return analysis_result
        
    finally:
        # Clean up the temporary zip file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/api/train-model/")
async def train_model(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    task_type: str = Form(...)
):
    """
    Receives the dataset, target column, and task type, 
    then triggers the ML engine to train and evaluate algorithms.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
        
    temp_file_path = f"temp_train_{file.filename}"
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Trigger the ML Engine!
        training_results = train_and_evaluate(temp_file_path, target_column, task_type)
        
        if training_results.get("status") == "error":
            raise HTTPException(status_code=500, detail=training_results.get("message"))
            
        return training_results
        
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# --- NEW: Vision Training Endpoint ---
@app.post("/api/train-vision-model/")
async def train_vision_model_endpoint(file: UploadFile = File(...)):
    """
    Receives a ZIP file and triggers the Vision Engine to train Image Classifiers.
    """
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Only ZIP files are supported for Vision training.")
        
    temp_file_path = f"temp_train_vision_{file.filename}"
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Trigger the Vision Engine!
        training_results = train_vision_model(temp_file_path)
        
        if training_results.get("status") == "error":
            raise HTTPException(status_code=500, detail=training_results.get("message"))
            
        return training_results
        
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.get("/api/download-model/{filename}")
async def download_model(filename: str):
    """
    Endpoint to allow users to download their trained .joblib models.
    """
    file_path = os.path.join("saved_models", filename)
    
    # Security/Sanity check: Make sure the file actually exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Model file not found. It may have been deleted or never saved.")
        
    # Return the file as a downloadable attachment
    return FileResponse(
        path=file_path, 
        filename=filename, 
        media_type='application/octet-stream'
    )