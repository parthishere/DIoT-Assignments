import numpy as np
import joblib

def load_model(model_path):
    """Load a TensorFlow model."""
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
    input_array = np.array(data['features']).reshape(1, -1)
    return input_array

def postprocess_output(prediction):
    """Process model output to return readable results."""
    class_idx = np.argmax(prediction, axis=1)[0]
    class_names = ['class1', 'class2', 'class3'] 
    return class_names[class_idx]