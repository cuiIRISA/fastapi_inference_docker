model_name = "rtdetr"
instance_type = "ml.g5.2xlarge"


# Create model
print(f"Creating model: {model_name}")
model_response = sm_client.create_model(
    ModelName=model_name,
    PrimaryContainer={
        'Image': image_uri,
        'Mode': 'SingleModel',
    },
    ExecutionRoleArn=role_arn
)
print(f"Model ARN: {model_response['ModelArn']}")


# Create endpoint configuration
print(f"Creating endpoint configuration: {model_name}-config")
endpoint_config_response = sm_client.create_endpoint_config(
    EndpointConfigName=f"{model_name}-config",
    ProductionVariants=[
        {
            'VariantName': 'AllTraffic',
            'ModelName': model_name,
            'InstanceType': instance_type,
            'InitialInstanceCount': 1,
        }
    ]
)
print(f"Endpoint config ARN: {endpoint_config_response['EndpointConfigArn']}")

endpoint_name = "deployment-rtdetr"

# Create endpoint
print(f"Creating endpoint: {endpoint_name}")
endpoint_response = sm_client.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=f"{model_name}-config"
)
print(f"Endpoint ARN: {endpoint_response['EndpointArn']}")

# Wait for endpoint to be in service
print("Waiting for endpoint to be in service...")


import base64
image_path= "./id.png"

def encode_image_to_base64(image_path):
    """Encode an image file to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')  
    

session = boto3.Session()
runtime = session.client('sagemaker-runtime')


payload = {
    "image": encode_image_to_base64(image_path),
    "confidence_threshold": 0.3,
    "visualize": False
}

# Convert the payload dictionary to a JSON string
body = json.dumps(payload)

response = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType='application/json',
    Body=body  # Pass the JSON string, not the Python dictionary
)

result = json.loads(response['Body'].read().decode())
print(json.dumps(result, indent=2))
