# ML Model Deployment API

This README provides comprehensive setup instructions for deploying a machine learning model API on Google Cloud Platform (GCP). The service allows users to upload pre-trained models, make predictions, and download models through a REST API.

## Project Overview

This project implements a web service with three API endpoints:
1. **Upload Model API**: Upload and persist a model in a GCP storage bucket
2. **Prediction API**: Send sample data and receive a classification/prediction result
3. **Download Model API**: Retrieve a model from GCP storage

## Table of Contents

- [Prerequisites](#prerequisites)
- [Local Environment Setup](#local-environment-setup)
- [GCP Setup](#gcp-setup)
  - [Creating a GCP Project](#creating-a-gcp-project)
  - [Creating a Service Account and Key](#creating-a-service-account-and-key)
  - [Creating a Storage Bucket](#creating-a-storage-bucket)
  - [Setting up a VM Instance](#setting-up-a-vm-instance)
  - [Configuring Firewall Rules](#configuring-firewall-rules)
- [Application Setup](#application-setup)
  - [Project Structure](#project-structure)
  - [Flask API Implementation](#flask-api-implementation)
  - [Deployment](#deployment)
- [Testing the API](#testing-the-api)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- Python 3.8 or higher
- Google Cloud Platform account
- Basic understanding of machine learning concepts
- Familiarity with Flask and RESTful APIs

## Local Environment Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/ml-model-api.git
   cd ml-model-api
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Train a model locally**:
   ```bash
   python train_model.py
   ```
   This will create a model file (e.g., `kmeans_model.joblib`) in your project directory.

## GCP Setup

### Creating a GCP Project

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Click on the project dropdown at the top of the page
3. Click "New Project"
4. Enter a project name (e.g., "diot-instance")
5. Click "Create"
6. Make note of your Project ID

### Creating a Service Account and Key

1. In the GCP Console, navigate to "IAM & Admin" > "Service Accounts"
2. Click "Create Service Account"
3. Enter a service account name (e.g., "diot-service")
4. Click "Create and Continue"
5. Assign the following roles:
   - Storage Admin (for bucket and object access)
   - Compute Admin (for VM management)
6. Click "Continue" and then "Done"
7. Locate your new service account in the list
8. Click the three dots menu and select "Manage keys"
9. Click "Add Key" > "Create new key"
10. Choose JSON as the key type
11. Click "Create"
12. Save the downloaded JSON key file (you'll need it later)

### Creating a Storage Bucket

1. In the GCP Console, navigate to "Cloud Storage" > "Buckets"
2. Click "Create Bucket"
3. Enter a globally unique bucket name (e.g., "diot-bucket")
4. Choose a region (us-central1)
5. Set "Access control" to "Fine-grained" (not public)
6. Leave other settings as default
7. Click "Create"

### Setting up a VM Instance

1. In the GCP Console, navigate to "Compute Engine" > "VM instances"
2. Click "Create Instance"
3. Configure your VM:
   - Name: diot-instance
   - Region: Choose a region (e.g., us-central1)
   - Machine type: e2-medium (2 vCPU, 4GB memory)
   - Boot disk: Ubuntu 20.04 LTS
   - Firewall: Allow HTTP traffic
4. Click "Create"
5. Wait for the VM to initialize
6. Take note of the VM's external IP address

### Configuring Firewall Rules

1. In the GCP Console, navigate to "VPC Network" > "Firewall"
2. Click "Create Firewall Rule"
3. Configure the rule:
   - Name: allow-http
   - Network: default
   - Priority: 1000
   - Direction of traffic: Ingress
   - Action on match: Allow
   - Targets: All instances in the network
   - Source filter: IP ranges
   - Source IP ranges: 0.0.0.0/0 (allows from any IP)
   - Protocols and ports: TCP:80,22,1234,8000
4. Click "Create"

5. Create another rule to deny other incoming traffic:
   - Name: deny-all-incoming
   - Network: default
   - Priority: 2000 (higher number = lower priority)
   - Direction of traffic: Ingress
   - Action on match: Deny
   - Targets: All instances in the network
   - Source filter: IP ranges
   - Source IP ranges: 0.0.0.0/0
   - Protocols and ports: All
6. Click "Create"

## Application Setup

### Project Structure

Create the following directory structure on your VM:

```
ml-model-api/
├── app.py
├── requirements.txt
├── credentials.json
├── models/
├── downloads/
└── utils/
    ├── __init__.py
    ├── gcp_storage.py
    └── model_handler.py
```

### Flask API Implementation

1. **SSH into your VM**:
   ```bash
   gcloud compute ssh model-api-vm
   ```

2. **Install dependencies**:
   ```bash
   sudo apt update
   sudo apt install -y python3-pip python3-venv nginx
   ```

3. **Create project directory and files**:
   ```bash
   mkdir -p ml-model-api/models ml-model-api/downloads ml-model-api/utils
   cd ml-model-api
   touch app.py requirements.txt utils/__init__.py utils/gcp_storage.py utils/model_handler.py
   ```

4. **Upload your service account key file**:
   Copy your downloaded service account key to the VM as `credentials.json`.

5. **Install Python dependencies**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install flask flask-cors google-cloud-storage joblib scikit-learn numpy python-dotenv gunicorn
   pip freeze > requirements.txt
   ```

6. **Create `utils/gcp_storage.py`**:
   ```python
   from google.cloud import storage
   import os

   # Set credentials
   os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./credentials.json"

   def upload_blob(bucket_name, source_file_name, destination_blob_name):
       """Uploads a file to the bucket."""
       storage_client = storage.Client()
       bucket = storage_client.bucket(bucket_name)
       blob = bucket.blob(destination_blob_name)
       
       blob.upload_from_filename(source_file_name)
       
       return f"gs://{bucket_name}/{destination_blob_name}"

   def download_blob(bucket_name, blob_name, destination_file_name):
       """Downloads a blob from the bucket."""
       storage_client = storage.Client()
       bucket = storage_client.bucket(bucket_name)
       blob = bucket.blob(blob_name)
       
       os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
       blob.download_to_filename(destination_file_name)
       
       return destination_file_name
   ```

7. **Create `utils/model_handler.py`**:
   ```python
   import joblib
   import numpy as np

   def load_model(model_path):
       """Load a joblib model."""
       try:
           model = joblib.load(model_path)
           return model
       except Exception as e:
           raise Exception(f"Failed to load model: {str(e)}")

   def predict(model, data):
       """Make prediction using the model."""
       try:
           # Convert input data to appropriate format
           input_data = preprocess_input(data)
           
           # Make prediction
           prediction = model.predict(input_data)
           
           # Process output
           result = postprocess_output(prediction)
           
           return result
       except Exception as e:
           raise Exception(f"Prediction failed: {str(e)}")

   def preprocess_input(data):
       """Preprocess input data for model."""
       # For K-means clustering, we need x and y coordinates
       input_array = np.array(data['features']).reshape(1, -1)
       return input_array

   def postprocess_output(prediction):
       """Process model output to return readable results."""
       # For K-means, the prediction is directly the cluster number (an integer)
       cluster_id = int(prediction[0])
       return cluster_id
   ```

8. **Create `app.py`**:
   ```python
   from flask import Flask, jsonify, request, send_file
   from dotenv import load_dotenv
   import os
   from flask_cors import CORS
   import uuid
   import joblib

   from utils.model_handler import load_model, predict
   from utils.gcp_storage import upload_blob, download_blob

   load_dotenv()

   app = Flask(__name__)
   CORS(app)

   app.config['DEBUG'] = os.environ.get('FLASK_DEBUG', 'False') == 'True'

   # Global variable to store current model
   current_model = None
   current_model_path = None
   bucket_name = "your-bucket-name"  # Replace with your bucket name

   @app.route('/')
   def home():
       return "Model API Service"

   @app.route('/api/model/upload', methods=['POST'])
   def post_model_view():
       global current_model, current_model_path
       
       try:
           model_file = request.files.get('model')
           if not model_file:
               return jsonify({"error": "No model file provided"}), 400
               
           model_id = str(uuid.uuid4())
           model_filename = f"model_{model_id}"
           
           # Save the file locally
           temp_path = f"./models/{model_filename}"
           os.makedirs(os.path.dirname(temp_path), exist_ok=True)
           model_file.save(temp_path)
           
           # Load model into memory
           current_model = load_model(temp_path)
           current_model_path = temp_path
           
           # Upload to GCP bucket
           blob_name = f"models/{model_filename}"
           upload_blob(bucket_name, temp_path, blob_name)
           
           return jsonify({"success": True, "model_id": model_id}), 200
           
       except Exception as e:
           return jsonify({"error": str(e)}), 500

   @app.route('/api/predict', methods=['POST'])
   def predict_sample():
       global current_model
       
       # Check if model is loaded
       if current_model is None:
           return jsonify({"error": "No model is currently loaded"}), 400
           
       # Get input data
       data = request.get_json()
       if not data:
           return jsonify({"error": "No data provided"}), 400
           
       # Make prediction
       try:
           result = predict(current_model, data)
           return jsonify({"prediction": result}), 200
       except Exception as e:
           return jsonify({"error": f"Prediction error: {str(e)}"}), 500
           
   @app.route('/api/model/download', methods=['GET'])
   def download_model():
       global current_model_path
       
       # Check if model exists
       if current_model_path is None:
           return jsonify({"error": "No model is currently loaded"}), 400
           
       try:
           # Get the blob name from the current model path
           model_filename = os.path.basename(current_model_path)
           blob_name = f"models/{model_filename}"
           
           # Create a downloads directory if it doesn't exist
           download_dir = "./downloads"
           os.makedirs(download_dir, exist_ok=True)
           
           # Set the download path
           download_path = f"{download_dir}/{model_filename}"
           
           # Download from GCP bucket
           download_blob(bucket_name, blob_name, download_path)
           
           # Return the downloaded file
           return send_file(
               download_path,
               as_attachment=True,
               download_name=model_filename
           )
       except Exception as e:
           return jsonify({"error": f"Download error: {str(e)}"}), 500
       
   if __name__ == "__main__":
       app.run(host='0.0.0.0', port=8000)
   ```

9. **Create a `.env` file** (optional):
   ```
   FLASK_DEBUG=False
   ```

### Deployment

1. **Configure Nginx as a reverse proxy**:
   ```bash
   sudo nano /etc/nginx/sites-available/model-api
   ```

   Add the following configuration:
   ```
   server {
       listen 80;
       server_name _;

       location / {
           proxy_pass http://localhost:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   ```

2. **Enable the site and restart Nginx**:
   ```bash
   sudo ln -s /etc/nginx/sites-available/model-api /etc/nginx/sites-enabled/
   sudo rm /etc/nginx/sites-enabled/default  # Remove default site
   sudo systemctl restart nginx
   ```


## Testing the API

Create a test script (`test_api.py`) to test all the APIs:

```python
import requests
import json
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Configuration
BASE_URL = "http://your-vm-ip"  # Change to your VM's IP address
UPLOAD_ENDPOINT = f"{BASE_URL}/api/model/upload"
PREDICT_ENDPOINT = f"{BASE_URL}/api/predict"
DOWNLOAD_ENDPOINT = f"{BASE_URL}/api/model/download"

def test_local_model():
    """Train a K-means model locally and save it using joblib"""
    print("Training local model...")
    
    # Load and prepare the data
    df = pd.read_csv('your_dataset.csv')  # Replace with your dataset path
    X = df[['x', 'y']].values  # Extract features (x, y coordinates)
    
    # Train K-means model
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    
    # Save the model using joblib
    model_path = 'kmeans_model.joblib'
    joblib.dump(kmeans, model_path)
    print(f"Model saved to {model_path}")
    
    # Visualize the clusters (optional)
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis', s=50, alpha=0.8)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                c='red', marker='X', s=200)
    plt.title('K-means Clustering Result')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('clustering_result.png')
    plt.close()
    
    return model_path

def test_upload_model(model_path):
    """Test the model upload API endpoint"""
    print("\nTesting model upload API...")
    
    # Create a multipart form request
    with open(model_path, 'rb') as f:
        files = {'model': (os.path.basename(model_path), f, 'application/octet-stream')}
        response = requests.post(UPLOAD_ENDPOINT, files=files)
    
    print(f"Status code: {response.status_code}")
    
    try:
        json_response = response.json()
        print(f"Response: {json_response}")
        
        if response.status_code == 200:
            print("✅ Model upload successful")
            return True
        else:
            print("❌ Model upload failed")
            return False
    except requests.exceptions.JSONDecodeError:
        print(f"Response text: {response.text[:200]}...")
        print("❌ Response is not valid JSON")
        return False

def test_predict(sample_points):
    """Test the prediction API endpoint with multiple samples"""
    print("\nTesting prediction API...")
    
    for i, point in enumerate(sample_points):
        # Create sample data in the correct format
        sample_data = {
            'features': point.tolist()  # Convert numpy array to list
        }
        
        # Send prediction request
        response = requests.post(
            PREDICT_ENDPOINT, 
            headers={'Content-Type': 'application/json'},
            data=json.dumps(sample_data)
        )
        
        print(f"\nSample {i+1} - Point {point}:")
        print(f"Status code: {response.status_code}")
        
        try:
            json_response = response.json()
            print(f"Response: {json_response}")
            
            if response.status_code == 200:
                print(f"✅ Prediction successful - Cluster: {json_response.get('prediction')}")
            else:
                print("❌ Prediction failed")
        except requests.exceptions.JSONDecodeError:
            print(f"Response text: {response.text[:200]}...")
            print("❌ Response is not valid JSON")

def test_download_model():
    """Test the model download API endpoint"""
    print("\nTesting model download API...")
    
    # Send download request
    response = requests.get(DOWNLOAD_ENDPOINT)
    
    print(f"Status code: {response.status_code}")
    
    if response.status_code == 200:
        # Save the downloaded model
        downloaded_model_path = 'downloaded_model.joblib'
        with open(downloaded_model_path, 'wb') as f:
            f.write(response.content)
        
        print(f"✅ Model download successful - Saved to {downloaded_model_path}")
        
        # Verify the model works
        try:
            model = joblib.load(downloaded_model_path)
            print(f"Model loaded successfully: {type(model)}")
            return True
        except Exception as e:
            print(f"❌ Error loading downloaded model: {str(e)}")
            return False
    else:
        try:
            json_response = response.json()
            print(f"Error: {json_response}")
        except:
            print(f"Response: {response.text[:100]}...")
        return False

def main():
    """Main test function"""
    print("=== API Testing Script for K-means Clustering Model ===")
    
    # Step 1: Train and save model locally
    model_path = test_local_model()
    
    # Step 2: Test model upload
    upload_success = test_upload_model(model_path)
    
    if upload_success:
        # Step 3: Test predictions with sample points
        # Create a few test points that represent different areas of your feature space
        test_points = np.array([
            [-8.5, -5.6],    # likely cluster 0
            [-11.0, -9.0],   # likely cluster 1
            [-1.7, 10.5]     # likely cluster 2
        ])
        test_predict(test_points)
        
        # Step 4: Test model download
        download_success = test_download_model()
        
        if download_success:
            print("\n✅ All API tests completed successfully")
        else:
            print("\n⚠️ Download test failed, but upload and predict tests completed")
    else:
        print("\n❌ Upload test failed, skipping prediction and download tests")

if __name__ == "__main__":
    main()
```

Replace `your-vm-ip` with your VM's external IP address, and `your_dataset.csv` with your actual dataset file.

## Troubleshooting

### Permission Issues

If you encounter 403 Forbidden errors with GCP Storage:
1. Verify your service account has the correct roles
2. Check that your credentials.json file is correct and accessible
3. Verify the bucket exists and is in the same project

### Connection Issues

If you can't connect to your API:
1. Check if your VM is running
2. Verify firewall rules are set up correctly
3. Ensure Nginx is configured properly and running
4. Check if your Flask app is running (systemctl status model-api)

### API Errors

For errors in API responses:
1. Check the logs: `sudo journalctl -u model-api -f`
2. Verify the model format is compatible with your code
3. Check for any syntax errors in your Python files

### Missing Dependencies

If your app is failing due to missing dependencies:
1. Activate your virtual environment: `source venv/bin/activate`
2. Install any missing packages: `pip install <package-name>`
3. Update requirements.txt: `pip freeze > requirements.txt`