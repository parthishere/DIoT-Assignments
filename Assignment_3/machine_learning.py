import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import joblib, os

filename = 'network_model.pkl'


def get_model():
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        try:
            model = joblib.load(filename)
        except Exception as e:
            print(e)
    else:
        model = False
        print(f"File is empty or does not exists : {filename}")
        
    return model

def read_and_visualize_data():
    pdata = pd.read_csv("./train.csv")
    print("\n====== train.csv info ======")
    print(pdata.info())

    print("\n====== service values ======")
    print(pdata.service.value_counts())
    print("\n====== lable values ======")
    print(pdata.label.value_counts())
    print("\n====== protocol_types values ======")
    print(pdata.protocol_type.value_counts())
    print("\n====== data.head() ======")
    print(pdata.head())
    print("\n====== data.tail() ======")
    print(pdata.tail())
    print("\n====== data.shape ======")
    print(pdata.shape)
    return pdata

def cleanup_data(pdata):
    
    cat_features = ['protocol_type', 'service', 'flag']
    encoders = {}
    for feature in cat_features:
        le = LabelEncoder()
        pdata[feature] = le.fit_transform(pdata[feature])
        encoders[feature] = le
        print(f"\n{feature} mapping:")
        for i, label in enumerate(le.classes_):
            print(f"  {label} -> {i}")


    label_encoder = LabelEncoder()
    pdata['label'] = label_encoder.fit_transform(pdata['label'])
    
    
    X = pdata.drop('label', axis=1)
    y = pdata['label']
    
    numerical_features = [col for col in X.columns if col not in cat_features]

    # First, replace 'Missing' strings with NaN
    for col in numerical_features:
        if X[col].dtype == 'O':  # If column is of object type (strings)
            X[col] = X[col].replace('Missing', np.nan)
            
        # Convert all values to float, coercing errors to NaN
        X[col] = pd.to_numeric(X[col], errors='coerce')

    # Option 1: Fill with mean
    X[numerical_features] = X[numerical_features].fillna(X[numerical_features].mean())
    # Option 2 (alternative): Fill with median (more robust to outliers)
    # X[numerical_features] = X[numerical_features].fillna(X[numerical_features].median())
    # Option 3 (alternative): Fill with 0
    # X[numerical_features] = X[numerical_features].fillna(0)



    scaler = StandardScaler()
    numerical_features = [col for col in X.columns if col not in cat_features]
    X[numerical_features] = scaler.fit_transform(X[numerical_features])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    num_classes = len(label_encoder.classes_)
    y_train_categorical = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test_categorical = tf.keras.utils.to_categorical(y_test, num_classes)
    
    return X_train, X_test, y_train, y_test, y_train_categorical, y_test_categorical, num_classes, label_encoder



def create_and_save_model( X_train, X_test, y_train, y_test, y_train_categorical, y_test_categorical, num_classes):
    
    keras.backend.clear_session() # clearing session
    np.random.seed(42) # generating random see
    tf.random.set_seed(42) # set.seed function helps reuse same set of random variables
    
    

    model = keras.models.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    print("\nModel Summary:")
    model.summary()

    print("\nTraining the model...")
    history = model.fit(
        X_train, y_train_categorical,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    joblib.dump(model, filename)
    return model
        

def evaluate_model(model, X_test, y_test, y_test_categorical, label_encoder):
    
 
    print("\nEvaluating the model...")
    loss, accuracy = model.evaluate(X_test, y_test_categorical)
    print(f"Test Accuracy: {accuracy:.4f}")

    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    print("\nSample predictions:")
    for i in range(min(5, len(predictions))):
        actual_class = label_encoder.inverse_transform([y_test.iloc[i]])[0]
        predicted_class = label_encoder.inverse_transform([predicted_classes[i]])[0]
        print(f"Sample {i+1}: Actual: {actual_class}, Predicted: {predicted_class}")
        
        
        
def generate_submission_file(model, label_encoder):

    print("\nLoading test dataset...")
    test_data = pd.read_csv("./test.csv")

    ids = test_data['id'].values
    
    cat_features = ['protocol_type', 'service', 'flag']
    for feature in cat_features:
        le = LabelEncoder()
        test_data[feature] = le.fit_transform(test_data[feature])
    
    X_test = test_data.drop('id', axis=1)
    
    numerical_features = [col for col in X_test.columns if col not in cat_features]
    for col in numerical_features:
        if X_test[col].dtype == 'O':  # If column is of object type (strings)
            X_test[col] = X_test[col].replace('Missing', np.nan)
        
        # Convert all values to float, coercing errors to NaN
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

    X_test[numerical_features] = X_test[numerical_features].fillna(X_test[numerical_features].mean())

    scaler = StandardScaler()
    X_test[numerical_features] = scaler.fit_transform(X_test[numerical_features])

    print("Making predictions on test dataset...")
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    
    predicted_labels = label_encoder.inverse_transform(predicted_classes)
    
    submission = pd.DataFrame({
        'Id': ids,
        'Category': predicted_labels
    })
    
    submission_filename = 'submission.csv'
    submission.to_csv(submission_filename, index=False)
    print(f"Submission file saved as '{submission_filename}'")
    print(f"Total predictions: {len(submission)}")
    print("\nSample of submission file:")
    print(submission.head())
    
    
if __name__ == '__main__':
    model = get_model()
    pddata = read_and_visualize_data()
    X_train, X_test, y_train, y_test, y_train_categorical, y_test_categorical, num_classes, label_encoder = cleanup_data(pddata)
    if model:
        print("Loaded Existing model, now running evulation!")
        evaluate_model(model, X_test, y_test, y_test_categorical, label_encoder)
        generate_submission_file(model, label_encoder)
    else:
        print("Creating and Saving Neural networks!")
        model = create_and_save_model(X_train, X_test, y_train, y_test, y_train_categorical, y_test_categorical, num_classes)
        evaluate_model(model, X_test, y_test, y_test_categorical, label_encoder)
        generate_submission_file(model, label_encoder)
    print("Finished")
    


    