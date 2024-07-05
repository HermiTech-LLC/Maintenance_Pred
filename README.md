# Predictive Maintenance Model

This script (`mainPre.py`) trains a neural network model to predict equipment failure based on sensor data.

## Features

- Handles missing values
- Feature engineering capabilities including interaction terms and polynomial features
- Model training with early stopping to prevent overfitting
- Model evaluation with accuracy, loss, classification report, and confusion matrix
- Model and scaler saving for deployment
- Visualization of training history

## Requirements

- TensorFlow
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Joblib

## Installation

Install the required libraries:
```
pip install tensorflow numpy pandas scikit-learn matplotlib joblib
```

## Usage

1. Update the `data_url` variable in the script to point to your dataset.

2. Run the script:
```
python mainPre.py
```

3. The trained model will be saved as `predictive_maintenance_model.h5`.
4. The scaler will be saved as `scaler.pkl`.
5. Training history plots will be saved as `training_history.png` and `accuracy_history.png`.

## Data Preparation

- Ensure the dataset is in CSV format.
- The script assumes the dataset contains columns named `sensor1`, `sensor2`, `sensor3`, `sensor4`, and `sensor5` for sensor data, and `failure` as the target variable.
- Adjust the column names in the script as necessary to match your dataset.

## Model Deployment

Load the saved model and scaler in your deployment script:
```python
from tensorflow.keras.models import load_model
import joblib
import numpy as np

model = load_model('predictive_maintenance_model.h5')
scaler = joblib.load('scaler.pkl')

def predict_failure(sensor_data):
    sensor_data = np.array(sensor_data).reshape(1, -1)
    sensor_data = scaler.transform(sensor_data)
    prediction = model.predict(sensor_data)
    return prediction > 0.5

# Example usage
sensor_data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.02, 0.09]  # Replace with actual sensor data
print(predict_failure(sensor_data))
```

## License

BSD 3-Clause License
