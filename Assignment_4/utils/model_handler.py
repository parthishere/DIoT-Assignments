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
    # No need for argmax since K-means.predict() already returns the cluster index
    cluster_id = int(prediction[0])
    return cluster_id