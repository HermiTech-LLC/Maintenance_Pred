import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load and preprocess the data
data_url = 'datasets/predictive_maintenance.csv'  # Replace with actual dataset path
if not os.path.exists(data_url):
    logger.error(f"Dataset not found at path: {data_url}")
    raise FileNotFoundError(f"Dataset not found at path: {data_url}")
data = pd.read_csv(data_url)
logger.info("Data loaded successfully")

# Preprocessing steps
sensor_columns = ['sensor1', 'sensor2', 'sensor3', 'sensor4', 'sensor5']  # Replace with actual sensor columns
target_column = 'failure'  # Binary target variable indicating equipment failure

# Handle missing values
data = data.dropna()
logger.info("Missing values handled")

# Feature engineering: Add interaction terms or domain-specific features if needed
data['sensor1_sensor2_interaction'] = data['sensor1'] * data['sensor2']
data['sensor3_squared'] = data['sensor3'] ** 2

# Update sensor columns after feature engineering
sensor_columns = ['sensor1', 'sensor2', 'sensor3', 'sensor4', 'sensor5', 'sensor1_sensor2_interaction', 'sensor3_squared']

X = data[sensor_columns]
y = data[target_column]

# Handle class imbalance
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)
logger.info("Class imbalance handled using SMOTE")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
logger.info("Data split into training and test sets")

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
logger.info("Features normalized")

# Hyperparameter tuning with GridSearchCV and cross-validation
def build_model(optimizer='adam', dropout_rate=0.0):
    model = Sequential([
        Dense(32, input_shape=(X_train.shape[1],), activation='relu'),
        Dropout(dropout_rate),
        Dense(16, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=build_model, verbose=0)

param_grid = {
    'optimizer': ['adam', 'rmsprop'],
    'dropout_rate': [0.0, 0.2, 0.5],
    'epochs': [50, 100],
    'batch_size': [16, 32]
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, n_jobs=-1, scoring='accuracy')
grid_result = grid.fit(X_train, y_train)
logger.info(f"Best parameters found: {grid_result.best_params_}")

# Build and compile the final model with best parameters
best_params = grid_result.best_params_
model = build_model(optimizer=best_params['optimizer'], dropout_rate=best_params['dropout_rate'])

# Add early stopping and model checkpoint to save the best model
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')

# Train the final model
history = model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], validation_split=0.2, callbacks=[early_stopping, model_checkpoint])
logger.info("Model training complete")

# Load the best model
model = tf.keras.models.load_model('best_model.h5')

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
logger.info(f'Accuracy: {accuracy:.2f}, Loss: {loss:.2f}')

# Generate classification report and confusion matrix
y_pred = (model.predict(X_test) > 0.5).astype(int)
logger.info(f"\nClassification Report:\n {classification_report(y_test, y_pred)}")
logger.info(f"\nConfusion Matrix:\n {confusion_matrix(y_test, y_pred)}")
logger.info(f'ROC AUC Score: {roc_auc_score(y_test, y_pred):.2f}')

# Save the final model and scaler
model.save('predictive_maintenance_model.h5')
joblib.dump(scaler, 'scaler.pkl')
logger.info("Model and scaler saved")

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
