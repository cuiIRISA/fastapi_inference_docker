#!/usr/bin/env python3
"""
Improved parallel load testing script for SageMaker RF-DETR endpoint
with enhanced logging and concurrent execution monitoring
"""
import argparse
import json
import base64
import time
import datetime
import concurrent.futures
import boto3
from pathlib import Path
from botocore.exceptions import ClientError
from botocore.config import Config
import threading
import uuid

# Track requests with thread-local storage
thread_local = threading.local()

def get_boto3_client(region):
    """Get thread-local boto3 client for connection pooling"""
    if not hasattr(thread_local, "client"):
        # Configure boto3 with retries and connection pooling
        config = Config(
            retries={"max_attempts": 3, "mode": "standard"},
            max_pool_connections=20
        )
        session = boto3.Session(region_name=region)
        thread_local.client = session.client('sagemaker-runtime', config=config)
    return thread_local.client

def encode_image_to_base64(image_path):
    """Encode an image file to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_timestamp():
    """Get current timestamp for logging"""
    return datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]

def send_request(image_path, endpoint_name, region, confidence=0.25, worker_id=None):
    """Send inference request to SageMaker endpoint"""
    request_id = worker_id or uuid.uuid4().hex[:8]
    start_time_log = time.time()
    log_prefix = f"[{get_timestamp()}][Worker {request_id}]"
    
    try:
        print(f"{log_prefix} Starting request")
        
        # Use connection pooling with thread-local client
        runtime = get_boto3_client(region)
        
        # Cache encoded image if not already done (optimization for multiple iterations)
        if not hasattr(thread_local, "image_cache") or thread_local.image_path != image_path:
            thread_local.image_path = image_path
            print(f"{log_prefix} Encoding image")
            thread_local.image_cache = encode_image_to_base64(image_path)
        
        image_b64 = thread_local.image_cache
        
        payload = {
            "image": image_b64,
            "confidence_threshold": confidence,
            "visualize": False
        }

        # Send request and measure precise timing
        print(f"{log_prefix} Sending request to endpoint")
        start_time = time.time()
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        end_time = time.time()
        latency = end_time - start_time
        
        # Process response
        result = json.loads(response['Body'].read().decode())
        detected_objects = len(result.get('detections', [])) if 'detections' in result else 0
        
        print(f"{log_prefix} Request completed in {latency:.3f}s with {detected_objects} objects detected")
            
        return {
            "worker_id": request_id,
            "success": True,
            "latency": latency,
            "timestamp": {
                "start": start_time,
                "end": end_time
            },
            "objects_detected": detected_objects,
            "total_time": time.time() - start_time_log
        }

    except Exception as e:
        error_time = time.time()
        print(f"{log_prefix} Request failed: {e}")
        return {
            "worker_id": request_id,
            "success": False,
            "error": str(e),
            "timestamp": {
                "start": start_time_log,
                "error": error_time
            },
            "latency": error_time - start_time_log if 'start_time' in locals() else 0
        }

def run_load_test(image_path, endpoint_name, region, workers=10, iterations=5, confidence=0.25):
    """Execute load test with true parallel workers"""
    total_requests = workers * iterations
    
    # Print load test configuration
    print(f"\n{'='*50}")
    print(f"PARALLEL LOAD TEST - {get_timestamp()}")
    print(f"{'='*50}")
    print(f"Workers: {workers} | Iterations per worker: {iterations} | Total requests: {total_requests}")
    print(f"Endpoint: {endpoint_name} | Region: {region}")
    print(f"Image: {image_path}")
    print(f"{'='*50}\n")

    # Track overall test metrics
    overall_start = time.time()
    results = []
    active_workers = []
    
    # Create a map of futures to their IDs
    future_to_id = {}
    
    # Configure thread pool with the requested number of workers
    # This ensures true parallelism up to the worker count
    print(f"[{get_timestamp()}] Initializing thread pool with {workers} workers")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks at once to maximize parallelism
        print(f"[{get_timestamp()}] Submitting {total_requests} requests to thread pool")
        
        # First submit all tasks to maximize concurrency
        for worker_id in range(workers):
            for i in range(iterations):
                request_id = f"{worker_id:02d}_{i:02d}"
                future = executor.submit(
                    send_request,
                    image_path=image_path,
                    endpoint_name=endpoint_name,
                    region=region,
                    confidence=confidence,
                    worker_id=request_id
                )
                future_to_id[future] = request_id
                active_workers.append(request_id)
                
        print(f"[{get_timestamp()}] All {total_requests} requests submitted")
        print(f"[{get_timestamp()}] Active concurrent workers: {len(active_workers)}")
        print(f"[{get_timestamp()}] Waiting for results...\n")
        
        # Track in-flight requests over time for concurrency visualization
        concurrency_datapoints = []
        last_log_time = time.time()
        
        # Process results as they complete (this is where parallelism happens)
        for idx, future in enumerate(concurrent.futures.as_completed(future_to_id)):
            request_id = future_to_id[future]
            result = future.result()
            results.append(result)
            
            # Track active workers for concurrency monitoring
            if request_id in active_workers:
                active_workers.remove(request_id)
            
            # Log concurrency metrics every second
            now = time.time()
            if now - last_log_time > 1.0:
                concurrency_datapoints.append({
                    "time": now - overall_start,
                    "active_requests": len(active_workers),
                    "completed": idx + 1
                })
                last_log_time = now
            
            completed_percent = (idx + 1) / total_requests * 100
            status = "✅" if result.get('success', False) else "❌"
            print(f"[{get_timestamp()}] {status} [{completed_percent:5.1f}%] Request {request_id} completed in {result.get('latency', 0):.3f}s")
        
    # Calculate test metrics
    overall_time = time.time() - overall_start
    successful = sum(1 for r in results if r.get('success', False))
    failed = total_requests - successful
    latencies = [r.get('latency', 0) for r in results if r.get('success', False)]
    
    # Print load test summary
    print(f"\n{'='*50}")
    print(f"LOAD TEST RESULTS - {get_timestamp()}")
    print(f"{'='*50}")
    print(f"Total time: {overall_time:.2f}s")
    print(f"Successful requests: {successful}/{total_requests} ({successful/total_requests*100:.1f}%)")
    print(f"Failed requests: {failed}/{total_requests} ({failed/total_requests*100:.1f}% if any)")
    
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        p50 = sorted(latencies)[len(latencies) // 2]
        p90 = sorted(latencies)[int(len(latencies) * 0.9)]
        p99 = sorted(latencies)[int(len(latencies) * 0.99)]
        
        print(f"\nLatency metrics (in seconds):")
        print(f"  Average: {avg_latency:.3f}")
        print(f"  Minimum: {min_latency:.3f}")
        print(f"  Maximum: {max_latency:.3f}")
        print(f"  P50: {p50:.3f}")
        print(f"  P90: {p90:.3f}")
        print(f"  P99: {p99:.3f}")
        
        print(f"\nThroughput:")
        print(f"  Requests per second: {successful/overall_time:.2f}")
        
    # Display parallelism visualization metrics
    if concurrency_datapoints:
        print(f"\nConcurrency metrics:")
        for point in concurrency_datapoints:
            print(f"  Time: {point['time']:.1f}s - Active: {point['active_requests']} - Completed: {point['completed']}")
    
    print(f"{'='*50}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parallel Load Test for SageMaker Endpoints')
    parser.add_argument('--image', type=str, default="id.png", help='Path to the image file')
    parser.add_argument('--confidence', type=float, default=0.3, help='Confidence threshold (0.0-1.0)')
    parser.add_argument('--endpoint-name', type=str, required=True, help='SageMaker endpoint name')
    parser.add_argument('--region', type=str, default='us-east-1', help='AWS region for SageMaker endpoint')
    parser.add_argument('--workers', type=int, default=10, help='Number of parallel workers for load testing')
    parser.add_argument('--iterations', type=int, default=5, help='Number of iterations per worker')
    parser.add_argument('--single', action='store_true', help='Run a single request instead of a load test')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not Path(args.image).exists():
        print(f"Error: Image not found at {args.image}")
        exit(1)
    
    if args.single:
        # Run single request
        print(f"Running single inference request to {args.endpoint_name}")
        result = send_request(
            image_path=args.image,
            endpoint_name=args.endpoint_name,
            region=args.region,
            confidence=args.confidence
        )
        if result['success'] and 'result' in result:
            print(json.dumps(result['result'], indent=2))
    else:
        # Run parallel load test
        run_load_test(
            image_path=args.image,
            endpoint_name=args.endpoint_name,
            region=args.region,
            workers=args.workers,
            iterations=args.iterations,
            confidence=args.confidence
        )
