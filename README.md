# RF-DETR Object Detection with FastAPI for SageMaker


## Local Development

For local development and testing:

```bash
# Build the local development image
docker build -f Dockerfile -t rf-detr-inference .

# Run the container
docker run --gpus all -p 8080:8080 rf-detr-inference
```
docker exec -it 5f5dc26d3472 /bin/bash


# Basic load test with defaults (10 workers, 5 iterations each)
python test_local.py --load-test --workers 10 --iterations 5 --image ./id.png --url http://localhost:8080

# Customize worker count and iterations on SageMaker 
python parallel_load_test_sagemaker.py --endpoint-name "deployment-rtdetr-gpu"  --workers 30 --iterations 15

