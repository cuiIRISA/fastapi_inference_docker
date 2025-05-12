import os
import json
import base64
from io import BytesIO

from fastapi import FastAPI, Response, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

# Import the RF-DETR object detector
from rf_detr_detector import model_fn, transform_fn

# Initialize FastAPI for SageMaker deployment
app = FastAPI(
    title="RF-DETR Object Detection API for SageMaker",
    description="SageMaker-compatible API for object detection using RF-DETR model",
    version="1.0.0",
)

# Global variable for model
model = None

def load_model():
    """Load the model if not already loaded"""
    global model
    if model is None:
        try:
            # SageMaker expects models in /opt/ml/model
            model_dir = os.environ.get('MODEL_DIR', '/opt/ml/model')
            
            # Check if model directory exists
            if not os.path.exists(model_dir):
                raise FileNotFoundError(f"Model directory not found: {model_dir}")
                
            # Get the model file name from environment variable or use default
            model_file_name = os.environ.get('MODEL_FILE_NAME', 'rtdetr-l.pt')
            model_file_path = os.path.join(model_dir, model_file_name)
            
            # Log the paths for debugging
            print(f"Looking for model in directory: {model_dir}")
            print(f"Expected model file path: {model_file_path}")
            
            # List files in model directory for debugging
            if os.path.exists(model_dir):
                print(f"Files in model directory: {os.listdir(model_dir)}")
            
            # Load the model
            model = model_fn(model_dir)
            print(f"Model loaded successfully from {model_dir}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print(f"Error details: {type(e).__name__}")
            import traceback
            print(traceback.format_exc())
            raise
    return model

@app.on_event("startup")
async def startup_event():
    """Load the model on startup"""
    try:
        load_model()
    except Exception as e:
        print(f"Error loading model on startup: {str(e)}")
        # We don't want to stop the app from starting if model can't be loaded
        # It will attempt to load again on first request

@app.get('/ping')
def ping():
    """
    Health check endpoint required by SageMaker.
    SageMaker calls this to determine if the container is healthy.
    """
    try:
        model = load_model()
        return Response(content=json.dumps({"status": "ok"}), 
                      status_code=200, 
                      media_type="application/json")
    except Exception as e:
        return Response(content=json.dumps({"status": "error", "message": str(e)}), 
                      status_code=500, 
                      media_type="application/json")

@app.post('/invocations')
async def invocations(request: Request):
    """
    Inference endpoint required by SageMaker.
    Handles incoming requests for object detection.
    """
    # Load model
    try:
        model = load_model()
    except Exception as e:
        return Response(content=json.dumps({"error": f"Model loading failed: {str(e)}"}), 
                      status_code=503, 
                      media_type="application/json")
    
    # Get content type and accept type from headers
    content_type = request.headers.get('Content-Type', '')
    accept_type = request.headers.get('Accept', 'application/json')
    
    try:
        # Get the raw data
        data = await request.body()
        
        # Process the request using the transform function
        output, output_content_type = await transform_fn(model, data, content_type, accept_type)
        
        # Return appropriate response based on content type
        if output_content_type == 'application/octet-stream':
            return StreamingResponse(BytesIO(output), media_type=output_content_type)
        else:
            return Response(content=output, media_type=output_content_type)
            
    except Exception as e:
        return Response(content=json.dumps({"error": str(e)}), 
                      status_code=500, 
                      media_type="application/json")

if __name__ == '__main__':
    # The container runs on port 8080 in SageMaker
    uvicorn.run("sagemaker_serve:app", host="0.0.0.0", port=8080)
