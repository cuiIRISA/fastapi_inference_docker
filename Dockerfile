# Start with the PyTorch base image
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies - only what's needed for inference
# Note: torch and torchvision are already in the base image, so we can remove those
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    gunicorn \
    ultralytics \
    opencv-python \
    pillow \
    numpy \
    python-multipart

# SageMaker expects the container to have these directories
RUN mkdir -p /opt/ml/code /opt/ml/model

# Copy your code
COPY rf_detr_detector.py /opt/ml/code/
COPY sagemaker_serve.py /opt/ml/code/
COPY rtdetr-l.pt /opt/ml/model/

WORKDIR /opt/ml/code

# Expose SageMaker port
EXPOSE 8080

# Set Python path to include code directory
ENV PYTHONPATH=/opt/ml/code:${PYTHONPATH}

# Configure Gunicorn workers for parallel processing
ENV GUNICORN_WORKERS=4

# Use our production-ready server script for SageMaker
COPY serve /opt/ml/code/

# Make the script executable
RUN chmod +x /opt/ml/code/serve

# Start the SageMaker-compatible server
ENTRYPOINT ["python", "/opt/ml/code/serve"]
