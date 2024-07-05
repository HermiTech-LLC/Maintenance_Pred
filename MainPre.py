import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# Load and preprocess the data
data_url = 'datasets/predictive_maintenance.csv'  # Replace with actual dataset path
if not os.path.exists(data_url):
    raise FileNotFoundError(f"Dataset not found at path: {data_url}")
data = pd.read_csv(data_url)

# Preprocessing steps
sensor_columns = ['sensor1', 'sensor2', 'sensor3', 'sensor4', 'sensor5']  # Replace with actual sensor columns
target_column = 'failure'  # Binary target variable indicating equipment failure

# Handle missing values
data = data.dropna()

# Feature engineering: Add interaction terms or domain-specific features if needed
data['sensor1_sensor2_interaction'] = data['sensor1'] * data['sensor2']
data['sensor3_squared'] = data['sensor3'] ** 2

# Update sensor columns after feature engineering
sensor_columns = ['sensor1', 'sensor2', 'sensor3', 'sensor4', 'sensor5', 'sensor1_sensor2_interaction', 'sensor3_squared']

X = data[sensor_columns]
y = data[target_column]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the model
model = Sequential([
    Dense(32, input_shape=(X_train.shape[1],), activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Add early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy:.2f}, Loss: {loss:.2f}')

# Generate classification report and confusion matrix
y_pred = (model.predict(X_test) > 0.5).astype(int)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(f'ROC AUC Score: {roc_auc_score(y_test, y_pred):.2f}')

# Save the model
model.save('predictive_maintenance_model.h5')

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Visualize training history
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_history.png')
plt.show()

plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_history.png')
plt.show()
