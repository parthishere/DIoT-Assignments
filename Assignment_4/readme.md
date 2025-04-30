# ML Model Deployment API

setup instructions for deploying a machine learning model API on Google Cloud Platform (GCP). The service allows users to upload pre-trained models, make predictions, and download models through a REST API.

## Project Overview

This project implements a web service with three API endpoints:
1. **Upload Model API**: Upload and persist a model in a GCP storage bucket
2. **Prediction API**: Send sample data and receive a classification/prediction result
3. **Download Model API**: Retrieve a model from GCP storage

## Local Environment Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/parthishere/DIoT-Assignments
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
   This will create a model file (e.g., `kmeans_model.joblib`) in the directory.

## GCP Setup

### Creating a GCP Project

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Click on the project dropdown at the top of the page
3. Click "New Project"
4. Enter a project name (e.g., "diot-instance")
5. Click "Create"
6. Make note of the Project ID

### Creating a Service Account and Key

1. In the GCP Console, navigate to "IAM & Admin" > "Service Accounts"
2. Click "Create Service Account"
3. Enter a service account name (e.g., "diot-service")
4. Click "Create and Continue"
5. Assign the following roles:
   - Storage Admin (for bucket and object access)
   - Compute Admin (for VM management)
6. Click "Continue" and then "Done"
7. Locate the new service account in the list
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
3. Configure the VM:
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

Create the following directory structure on the VM:

```.
├── 2025-04-29 23-13-33.mkv
├── app.py
├── config.py
├── models
│   ├── data_K_Means_Clustering.csv
│   ├── Kmeans.ipynb
│   └── kmeans_model.joblib
├── readme.md
├── requirements.txt
├── test
│   ├── clustering_result.png
│   ├── downloaded_model.joblib
│   ├── kmeans_model.joblib
│   ├── test_output.txt
│   └── test.py
└── utils
    ├── gcp_storage.py
    ├── __init__.py
    ├── model_handler.py
    ├── __pycache__
    │   ├── gcp_storage.cpython-312.pyc
    │   ├── __init__.cpython-312.pyc
    │   └── model_handler.cpython-312.pyc
    └── service_account.json
```

### Flask API Implementation

1. **SSH into the VM**:
   ```bash
   gcloud compute ssh whatever
   ```

2. **Install dependencies**:
   ```bash
   sudo apt update
   sudo apt install -y python3-pip python3-venv nginx
   ```


3. **Upload the service account key file**:
   Copy the downloaded service account key to the VM as `service_account)credentials.json`.

5. **Install Python dependencies**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
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

Run test.py in test folder