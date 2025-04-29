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
bucket_name = "your-model-bucket"

@app.route('/')
def home():
    return "Model API Service"

@app.route('/api/post', methods=['POST'])
def post_model_view():
    global current_model, current_model_path;
    
    model_file = request.files.get('model');
    if not model_file:
        data = {'error': 'No model file provided'}
        return jsonify(data);
        
    model_id = str(uuid.uuid4())
    model_filename = f"model_{model_id}"
    current_model_path = model_filename
    
    # # Save the file locally as well
    temp_path = f"./models/{model_filename}"
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    joblib.dump(temp_path, model_file);
    
    # Upload to GCP bucket
    blob_name = f"models/{current_model_path}"
    upload_blob(bucket_name, current_model_path, blob_name)
    
    current_model = load_model(temp_path)
    

    return jsonify({"success": True, "model_id": model_id}), 200

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
        
    # Return model file
    try:
        blob_name = f"models/{current_model_path}"
        temp_path = f"./models/{current_model_path}"
        
        download_blob(bucket_name=bucket_name, blob_name=blob_name, destination_file_name=temp_path)
        return send_file(current_model_path, as_attachment=True)
    except Exception as e:
        return jsonify({"error": f"Download error: {str(e)}"}), 500
    
if __name__ == "__main__":
    app.run()


