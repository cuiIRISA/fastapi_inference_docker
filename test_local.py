#!/usr/bin/env python3
"""
Test script to demonstrate how to use the RF-DETR object detection API locally.
This version is specifically optimized for testing against a server already running with the serve script.

Usage:
    1. Command line: python test_local.py --image test_image.jpg [--visualize] [--confidence 0.25]
    2. Direct call: main(image_path="test_image.jpg", visualize=True, confidence=0.25, url="http://localhost:8080")
    3. Load testing: python test_local.py --image test_image.jpg --load-test --workers 10 --iterations 5
"""

import argparse
import requests
import json
import base64
import sys
import time
import datetime
import concurrent.futures
import threading
import uuid
from pathlib import Path

# Thread-local storage for optimizing resources across multiple requests
thread_local = threading.local()

def get_timestamp():
    """Get current timestamp for logging"""
    return datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]

def encode_image_to_base64(image_path):
    """Encode an image file to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def main(image_path=None, visualize=False, confidence=0.25, url="http://localhost:8080", worker_id=None):
    """
    Test the RF-DETR object detection API.
    
    Args:
        image_path (str): Path to the input image
        visualize (bool): Whether to visualize the detection results
        confidence (float): Confidence threshold for detections
        url (str): Base URL of the API
        
    Returns:
        dict: Detection results from the API
    """
    # Parse command line args if not provided as function parameters
    if image_path is None:
        parser = argparse.ArgumentParser(description='Test RF-DETR object detection API against serve deployment.')
        parser.add_argument('--image', type=str, required=True, help='Path to the image file')
        parser.add_argument('--visualize', action='store_true', help='Visualize detection results')
        parser.add_argument('--confidence', type=float, default=0.25, help='Confidence threshold (0.0-1.0)')
        parser.add_argument('--url', type=str, default='http://localhost:8080', 
                            help='Base URL of the API')
        args = parser.parse_args()
        
        image_path = args.image
        visualize = args.visualize
        confidence = args.confidence
        url = args.url
    
    # Convert to Path object
    image_path = Path(image_path)
    
    # Check if image exists
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        return None
        
    # Print info
    if worker_id is None:
        print(f"Testing RF-DETR object detection with:")
        print(f"  Image: {image_path}")
        print(f"  Confidence threshold: {confidence}")
        print(f"  Visualize: {visualize}")
        print(f"  API URL: {url}")
        print("-" * 40)
    else:
        print(f"Worker {worker_id}: Processing {image_path}")
    
    # Check if the server is ready
    try:
        response = requests.get(f"{url}/ping", timeout=5)
        if response.status_code == 200:
            print("Server is ready!")
        else:
            print(f"Warning: Server health check returned status {response.status_code}")
    except requests.RequestException as e:
        print(f"Warning: Could not connect to server: {e}")
        print("Continuing with inference request anyway...")
    
    # Send the inference request to the invocations endpoint (SageMaker compatible)
    try:
        # Choose between binary and JSON format
        if visualize:
            # Send binary image and get binary image back
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            print("Sending binary request to /invocations endpoint...")
            response = requests.post(
                f"{url}/invocations",
                data=image_data,
                headers={
                    "Content-Type": "application/octet-stream",
                    "Accept": "application/octet-stream"
                }
            )
            
        else:
            # Send JSON and get JSON back
            image_b64 = encode_image_to_base64(image_path)
            
            payload = {
                "image": image_b64,
                "confidence_threshold": confidence,
                "visualize": visualize
            }
            
            print("Sending JSON request to /invocations endpoint...")
            response = requests.post(
                f"{url}/invocations",
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
            )
            
    except Exception as e:
        print(f"Error during inference request: {e}")
        return None

    # Process the response
    result = None
    if response.status_code == 200:
        print(f"Success! Status code: {response.status_code}")
        
        # Check the content type
        content_type = response.headers.get('Content-Type', '')
        
        if 'image' in content_type or ('octet-stream' in content_type and visualize):
            # Save the image response with worker id if applicable
            output_suffix = f"_{worker_id}" if worker_id is not None else ""
            output_path = f"output_{image_path.stem}{output_suffix}.jpg"
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"Detection visualization saved to {output_path}")
            result = {"output_image": output_path}
            
        else:
            # Process the JSON response
            try:
                result = response.json()
                print("\nDetection Results:")
                print(f"Inference time: {result.get('inference_time', 'N/A')}s")
                
                detections = result.get('detections', [])
                print(f"Found {len(detections)} object(s)")
                
                for i, det in enumerate(detections, 1):
                    print(f"\nObject {i}:")
                    print(f"  Class: {det.get('class', 'unknown')}")
                    print(f"  Confidence: {det.get('confidence', 0):.4f}")
                    box = det.get('box', {})
                    print(f"  Bounding Box: [{box.get('x1', 0):.1f}, {box.get('y1', 0):.1f}] to [{box.get('x2', 0):.1f}, {box.get('y2', 0):.1f}]")
                
                # If there's a base64 image in the response, save it
                if visualize and 'image' in result:
                    try:
                        image_data = base64.b64decode(result['image'])
                        output_suffix = f"_{worker_id}" if worker_id is not None else ""
                        output_path = f"output_{image_path.stem}{output_suffix}.jpg"
                        with open(output_path, 'wb') as f:
                            f.write(image_data)
                        print(f"\nDetection visualization saved to {output_path}")
                        result["output_image"] = output_path
                    except Exception as e:
                        print(f"Error saving visualization: {e}")
                
            except Exception as e:
                print(f"Error parsing JSON response: {e}")
                print("Raw response:", response.text[:1000])  # Print first 1000 chars
                result = {"error": str(e), "raw_response": response.text[:1000]}
    else:
        print(f"Error: Status code {response.status_code}")
        print("Response:", response.text)
        result = {"error": f"Status code {response.status_code}", "response": response.text}
        
    return result

def worker_request(image_path, visualize=False, confidence=0.25, url="http://localhost:8080", worker_id=None):
    """
    Optimized worker function that only handles sending requests and processing responses.
    This function is designed for parallel execution.
    
    Args:
        image_path (str): Path to the input image
        visualize (bool): Whether to visualize the detection results
        confidence (float): Confidence threshold for detections
        url (str): Base URL of the API
        worker_id: Identifier for this worker
        
    Returns:
        dict: Result of the API call with timing information
    """
    request_id = worker_id or f"{uuid.uuid4().hex[:8]}"
    log_prefix = f"[{get_timestamp()}][Worker {request_id}]"
    start_time_total = time.time()
    
    try:
        print(f"{log_prefix} Starting request")
        
        # Cache encoded image if not already done (optimization for multiple iterations)
        if not hasattr(thread_local, "image_cache") or thread_local.image_path != image_path:
            thread_local.image_path = image_path
            print(f"{log_prefix} Encoding image")
            thread_local.image_cache = encode_image_to_base64(image_path)
            
        image_b64 = thread_local.image_cache
        
        payload = {
            "image": image_b64,
            "confidence_threshold": confidence,
            "visualize": visualize
        }
        
        # Send JSON and get JSON back
        print(f"{log_prefix} Sending request to API")
        start_time = time.time()
        
        response = requests.post(
            f"{url}/invocations",
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
        )
        
        end_time = time.time()
        latency = end_time - start_time
        
        if response.status_code == 200:
            # Process the JSON response
            try:
                result = response.json()
                detected_objects = len(result.get('detections', [])) if 'detections' in result else 0
                print(f"{log_prefix} Request completed in {latency:.3f}s with {detected_objects} objects detected")
                
                # If visualization was requested, save the image
                if visualize and 'image' in result:
                    try:
                        image_data = base64.b64decode(result['image'])
                        output_path = f"output_{Path(image_path).stem}_{request_id}.jpg"
                        with open(output_path, 'wb') as f:
                            f.write(image_data)
                        print(f"{log_prefix} Detection visualization saved to {output_path}")
                        result["output_image"] = output_path
                    except Exception as e:
                        print(f"{log_prefix} Error saving visualization: {e}")
                
                return {
                    "worker_id": request_id,
                    "success": True,
                    "latency": latency,
                    "inference_time": result.get('inference_time', 0),
                    "objects_detected": detected_objects,
                    "timestamp": {
                        "start": start_time,
                        "end": end_time
                    },
                    "total_time": time.time() - start_time_total,
                    "result": result
                }
            except Exception as e:
                print(f"{log_prefix} Error parsing JSON response: {e}")
                print(f"{log_prefix} Raw response: {response.text[:200]}...")
                return {
                    "worker_id": request_id,
                    "success": False,
                    "latency": latency,
                    "error": f"JSON parse error: {str(e)}",
                    "timestamp": {
                        "start": start_time,
                        "end": end_time
                    },
                    "total_time": time.time() - start_time_total
                }
        else:
            print(f"{log_prefix} Error: Status code {response.status_code}")
            return {
                "worker_id": request_id,
                "success": False,
                "latency": latency,
                "error": f"HTTP error: {response.status_code}",
                "response_text": response.text[:200],
                "timestamp": {
                    "start": start_time,
                    "end": end_time
                },
                "total_time": time.time() - start_time_total
            }
    
    except Exception as e:
        error_time = time.time()
        print(f"{log_prefix} Request failed with exception: {e}")
        return {
            "worker_id": request_id,
            "success": False,
            "error": str(e),
            "timestamp": {
                "start": start_time_total,
                "error": error_time
            },
            "total_time": error_time - start_time_total
        }

def run_load_test(image_path, num_workers, iterations, visualize=False, confidence=0.25, url="http://localhost:8080"):
    """
    Run a truly parallel load test with concurrent workers.
    
    Args:
        image_path (str): Path to the image file
        num_workers (int): Number of concurrent workers
        iterations (int): Number of iterations per worker
        visualize (bool): Whether to visualize results
        confidence (float): Confidence threshold
        url (str): API URL
    """
    total_requests = num_workers * iterations
    print(f"\n{'='*50}")
    print(f"PARALLEL LOAD TEST - {get_timestamp()}")
    print(f"{'='*50}")
    print(f"Workers: {num_workers} | Iterations per worker: {iterations} | Total requests: {total_requests}")
    print(f"Image: {image_path}")
    print(f"API URL: {url}")
    print(f"{'='*50}\n")
    
    # First check if server is running before starting load test
    try:
        response = requests.get(f"{url}/ping", timeout=5)
        if response.status_code == 200:
            print("Server is ready!")
        else:
            print(f"Warning: Server health check returned status {response.status_code}")
            print("Continuing with load test anyway...")
    except requests.RequestException as e:
        print(f"Warning: Could not connect to server: {e}")
        print("Load test may fail if the server is not running.")
        user_input = input("Do you want to continue anyway? (y/n): ")
        if user_input.lower() != 'y':
            print("Load test aborted.")
            return []
    
    overall_start = time.time()
    results = []
    active_workers = []
    
    # Map futures to their IDs
    future_to_id = {}
    
    # Configure thread pool with the requested number of workers
    print(f"[{get_timestamp()}] Initializing thread pool with {num_workers} workers")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks at once to maximize parallelism
        print(f"[{get_timestamp()}] Submitting {total_requests} requests to thread pool")
        
        # First submit all tasks to maximize concurrency
        for worker_id in range(num_workers):
            for i in range(iterations):
                request_id = f"{worker_id:02d}_{i:02d}"
                future = executor.submit(
                    worker_request,
                    image_path=image_path, 
                    visualize=visualize,
                    confidence=confidence,
                    url=url,
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
            try:
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
            except Exception as e:
                print(f"[{get_timestamp()}] ❌ Request {request_id} generated an exception: {e}")
    
    end_time = time.time()
    elapsed_time = end_time - overall_start
    
    # Calculate results
    successful = sum(1 for r in results if r.get('success', False))
    failed = total_requests - successful
    
    # Collect timing data
    latencies = [r.get('latency', 0) for r in results if r.get('success', False)]
    inference_times = [r.get('inference_time', 0) for r in results if r.get('success', False) and 'inference_time' in r]
    
    # Print load test summary
    print(f"\n{'='*50}")
    print(f"LOAD TEST RESULTS - {get_timestamp()}")
    print(f"{'='*50}")
    print(f"Total requests: {total_requests}")
    print(f"Successful requests: {successful}/{total_requests} ({successful/total_requests*100:.1f}%)")
    print(f"Failed requests: {failed}/{total_requests} ({failed/total_requests*100:.1f}% if any)")
    print(f"Total time: {elapsed_time:.2f} seconds")
    
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        p50 = sorted(latencies)[len(latencies) // 2]
        p90 = sorted(latencies)[int(len(latencies) * 0.9)]
        p99 = sorted(latencies)[int(len(latencies) * 0.99)]
        
        print(f"\nLatency metrics (in seconds):")
        print(f"  Average: {avg_latency:.4f}")
        print(f"  Minimum: {min_latency:.4f}")
        print(f"  Maximum: {max_latency:.4f}")
        print(f"  P50: {p50:.4f}")
        print(f"  P90: {p90:.4f}")
        print(f"  P99: {p99:.4f}")
        
        print(f"\nThroughput:")
        print(f"  Requests per second: {successful/elapsed_time:.2f}")
    
    # Calculate and print statistics if inference time is available
    if inference_times:
        avg_inference = sum(inference_times) / len(inference_times)
        max_inference = max(inference_times)
        min_inference = min(inference_times)
        print(f"\nInference time statistics:")
        print(f"  Average: {avg_inference:.4f} seconds")
        print(f"  Minimum: {min_inference:.4f} seconds")
        print(f"  Maximum: {max_inference:.4f} seconds")
    
    # Display parallelism visualization metrics
    if concurrency_datapoints:
        print(f"\nConcurrency metrics:")
        for point in concurrency_datapoints:
            print(f"  Time: {point['time']:.1f}s - Active: {point['active_requests']} - Completed: {point['completed']}")
    
    print(f"{'='*50}")
    return results

# Example of how to use this script directly by importing it
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test RF-DETR object detection API against serve deployment.')
    parser.add_argument('--image', type=str, required=False, default="static/bus.jpg", help='Path to the image file')
    parser.add_argument('--visualize', action='store_true', help='Visualize detection results')
    parser.add_argument('--confidence', type=float, default=0.25, help='Confidence threshold (0.0-1.0)')
    parser.add_argument('--url', type=str, default='http://localhost:8080', help='Base URL of the API')
    
    # Load test specific arguments
    parser.add_argument('--load-test', action='store_true', help='Run a load test with parallel requests')
    parser.add_argument('--workers', type=int, default=10, help='Number of parallel workers for load testing')
    parser.add_argument('--iterations', type=int, default=5, help='Number of iterations per worker')
    
    args = parser.parse_args()
    
    # Check if we should run a load test or a single request
    if args.load_test:
        results = run_load_test(
            image_path=args.image,
            num_workers=args.workers,
            iterations=args.iterations,
            visualize=args.visualize,
            confidence=args.confidence,
            url=args.url
        )
    else:
        # Run a single test
        result = main(
            image_path=args.image,
            visualize=args.visualize,
            confidence=args.confidence,
            url=args.url
        )
        
        print("\nExample complete. Result summary:")
        if result:
            if "output_image" in result:
                print(f"Image saved to: {result['output_image']}")
            elif "detections" in result:
                print(f"Detected {len(result['detections'])} objects")
        else:
            print("No result returned")
