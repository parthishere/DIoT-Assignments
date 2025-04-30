import requests
import json
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Configuration
BASE_URL = "http://35.202.18.79"  # Change to your VM's IP address
UPLOAD_ENDPOINT = f"{BASE_URL}/api/model/upload"
PREDICT_ENDPOINT = f"{BASE_URL}/api/predict"
DOWNLOAD_ENDPOINT = f"{BASE_URL}/api/model/download"

def test_local_model():
    """Train a K-means model locally and save it using joblib"""
    print("Training local model...")
    
    # Load and prepare the data
    df = pd.read_csv('../models/data_K_Means_Clustering.csv')  # Replace with your dataset path
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
    print(f"Response: {response.json()}")
    
    if response.status_code == 200:
        print("✅ Model upload successful")
        return True
    else:
        print("❌ Model upload failed")
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
        print(f"Response: {response.json()}")
        
        if response.status_code == 200:
            print(f"✅ Prediction successful - Cluster: {response.json().get('prediction')}")
        else:
            print("❌ Prediction failed")

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
        print("❌ Model download failed")
        if hasattr(response, 'json'):
            try:
                print(f"Error: {response.json()}")
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