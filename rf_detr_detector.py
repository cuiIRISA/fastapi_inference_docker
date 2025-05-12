import os
import json
import numpy as np
import cv2
import torch
from PIL import Image
import base64
import io
from typing import Dict, List, Union, Optional, Tuple
import time

class RFDETRObjectDetector:
    """
    RF-DETR Object Detection model implementation.
    """
    def __init__(self, model_path: str):
        """
        Initialize the RF-DETR model.
        
        Args:
            model_path: Path to the model weights/directory
        """
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.classes = self.model.names  # Class names from the model
        print(f"Model loaded successfully on {self.device}")
        print(f"Model supports {len(self.classes)} classes: {self.classes}")
        
    def _load_model(self):
        """
        Load the RF-DETR model.
        """
        try:
            # Check if the model path exists
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model path does not exist: {self.model_path}")
            
            print(f"Attempting to load model from: {self.model_path}")
            print(f"Using device: {self.device}")
            
            try:
                # Using ultralytics for RF-DETR loading
                from ultralytics import RTDETR
                print(f"Successfully imported ultralytics.RTDETR")
            except ImportError as e:
                print(f"Failed to import ultralytics.RTDETR: {e}")
                print("Checking if ultralytics package is installed...")
                try:
                    import pkg_resources
                    ultralytics_version = pkg_resources.get_distribution("ultralytics").version
                    print(f"Ultralytics package installed, version: {ultralytics_version}")
                except:
                    print("Unable to determine ultralytics version or package not installed.")
                raise
            
            # Load the model with appropriate parameters for RF-DETR
            print(f"Loading RTDETR model...")
            model = RTDETR(self.model_path)
            print(f"Model loaded, moving to device: {self.device}")
            model.to(self.device)
            print(f"Model successfully moved to device")
            
            # Validate the model was loaded correctly
            if not hasattr(model, 'names'):
                print(f"Warning: Model loaded but 'names' attribute is missing")
                # Set default class names if not available
                model.names = {0: 'object'}
                
            return model
        except Exception as e:
            print(f"Error loading RF-DETR model: {e}")
            import traceback
            print(traceback.format_exc())
            raise
    
    def predict(self, image_bytes: bytes, confidence_threshold: float = 0.25) -> Dict:
        """
        Perform object detection on the input image.
        
        Args:
            image_bytes: Raw image bytes
            confidence_threshold: Minimum confidence threshold for detections
            
        Returns:
            Dictionary with detection results
        """
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Start timing
            start_time = time.time()
            
            # Run inference
            results = self.model(image, conf=confidence_threshold)
            
            # End timing
            inference_time = time.time() - start_time
            
            # Process results
            detections = []
            for result in results:
                boxes = result.boxes
                
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].tolist()  # get box coordinates
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.classes[class_id]
                    
                    detection = {
                        "box": {
                            "x1": round(x1, 2),
                            "y1": round(y1, 2),
                            "x2": round(x2, 2),
                            "y2": round(y2, 2)
                        },
                        "confidence": round(confidence, 4),
                        "class_id": class_id,
                        "class": class_name
                    }
                    detections.append(detection)
            
            # Create response
            response = {
                "detections": detections,
                "inference_time": round(inference_time, 4),
                "image_width": image.width,
                "image_height": image.height
            }
            
            return response
        
        except Exception as e:
            print(f"Error during prediction: {e}")
            raise

    def visualize(self, image_bytes: bytes, detections: List[Dict]) -> bytes:
        """
        Draw bounding boxes on the image based on detections.
        
        Args:
            image_bytes: Raw image bytes
            detections: List of detection objects
            
        Returns:
            Annotated image bytes
        """
        try:
            # Convert to OpenCV format for drawing
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            for detection in detections:
                box = detection["box"]
                x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
                confidence = detection["confidence"]
                class_name = detection["class"]
                
                # Draw bounding box
                color = (0, 255, 0)  # Green color
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{class_name} {confidence:.2f}"
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x1, y1 - label_height - 5), (x1 + label_width, y1), color, -1)
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Convert back to bytes
            _, buffer = cv2.imencode('.jpg', img)
            return buffer.tobytes()
            
        except Exception as e:
            print(f"Error during visualization: {e}")
            raise

def encode_image(image_bytes):
    """
    Encode image bytes to base64 string.
    """
    return base64.b64encode(image_bytes).decode('utf-8')

# Functions that will be used by the FastAPI application

def model_fn(model_dir):
    """
    Load the object detection model.
    
    Args:
        model_dir: Directory containing the model files
        
    Returns:
        Loaded model object
    """
    try:
        # Get the model file name from environment variable or use default
        model_file_name = os.environ.get('MODEL_FILE_NAME', 'rf_detr_model.pt')
        model_file = os.path.join(model_dir, model_file_name)
        
        # Log paths for debugging
        print(f"Checking for model file: {model_file}")
        
        # Look for specific model files with common extensions if the specified one doesn't exist
        if not os.path.exists(model_file):
            print(f"Model file {model_file} not found, searching for alternatives...")
            common_extensions = ['.pt', '.pth', '.weights', '.onnx', '.bin']
            found_model = None
            
            # List all files in the model directory
            if os.path.isdir(model_dir):
                for file in os.listdir(model_dir):
                    file_path = os.path.join(model_dir, file)
                    if os.path.isfile(file_path) and any(file.endswith(ext) for ext in common_extensions):
                        found_model = file_path
                        print(f"Found alternative model file: {found_model}")
                        break
            
            # Use the found model or fall back to the directory itself
            model_path = found_model if found_model else model_dir
        else:
            model_path = model_file
            
        print(f"Using model path: {model_path}")
        
        # Validate the model path before loading
        if os.path.isdir(model_path):
            print(f"Model path is a directory: {model_path}")
            # Check if there are any model files in the directory
            model_files = [f for f in os.listdir(model_path) 
                          if os.path.isfile(os.path.join(model_path, f)) 
                          and any(f.endswith(ext) for ext in ['.pt', '.pth', '.weights', '.onnx', '.bin'])]
            if model_files:
                print(f"Found model files in directory: {model_files}")
            else:
                print(f"No model files found in directory: {model_path}")
                
        # Load the model
        model = RFDETRObjectDetector(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise

async def transform_fn(model, data, content_type, accept_type):
    """
    Process image data for object detection asynchronously.
    
    Args:
        model: Loaded model object
        data: Raw request data (image bytes or JSON)
        content_type: Content type of the request
        accept_type: Accepted response type
        
    Returns:
        tuple: (prediction result as JSON or image, content type)
    """
    try:
        # Parse the incoming request data
        confidence_threshold = 0.25  # Default confidence threshold
        visualize_results = False
        image_bytes = None
        
        if content_type.lower() == 'application/octet-stream':
            # Direct binary image data
            image_bytes = data
            
        elif content_type.lower() == 'application/json':
            # JSON request with potential parameters
            request_data = json.loads(data)
            
            if "image" in request_data:
                # Image comes as base64 encoded string
                image_bytes = base64.b64decode(request_data["image"])
                
            if "confidence_threshold" in request_data:
                confidence_threshold = float(request_data["confidence_threshold"])
                
            if "visualize" in request_data:
                visualize_results = bool(request_data["visualize"])
        else:
            raise ValueError(f"Unsupported content type: {content_type}")
        
        if image_bytes is None:
            raise ValueError("No image data found in request")
        
        # Direct GPU inference - no thread pool to avoid interfering with CUDA
        results = model.predict(image_bytes, confidence_threshold)
        
        # Return appropriate response based on accept_type and visualize parameter
        if visualize_results:
            # Direct visualization - consistent with the GPU inference approach
            visualized_image = model.visualize(image_bytes, results["detections"])
            
            if accept_type.lower() == 'application/octet-stream':
                return visualized_image, 'application/octet-stream'
            else:
                # Return base64 encoded image by default - direct encoding
                encoded_image = encode_image(visualized_image)
                response = {
                    "image": encoded_image,
                    "detections": results["detections"],
                    "inference_time": results["inference_time"]
                }
                return json.dumps(response), 'application/json'
        else:
            # Return just the detection results
            return json.dumps(results), 'application/json'
            
    except Exception as e:
        error = {"error": str(e)}
        return json.dumps(error), 'application/json'
