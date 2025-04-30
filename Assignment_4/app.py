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

app.config['DEBUG']  = os.environ.get('FLASK_DEBUG')

def upload():
    pass

# Global variable to store current model
current_model = None
current_model_path = None
bucket_name = "diot-ass4-bucket"

@app.route('/')
def home():
    return "Model API Service"

@app.route('/api/model/upload', methods=['POST'])
def post_model_view():
    global current_model, current_model_path;
    
    model_file = request.files.get('model');
    if not model_file:
        data = {'error': 'No model file provided'}
        return jsonify(data), 400
        
    model_id = str(uuid.uuid4())
    model_filename = f"model_{model_id}"
    
    # Save the file locally
    temp_path = f"./models/{model_filename}"
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    model_file.save(temp_path)
    
    try:
        # Upload to GCP bucket
        blob_name = f"models/{model_filename}"
        upload_blob(bucket_name, temp_path, blob_name)
        
        # Load model into memory
        current_model = load_model(temp_path)
        current_model_path = temp_path
        
        return jsonify({"success": True, "model_id": model_id}), 200
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500


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


