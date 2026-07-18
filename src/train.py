# src/train.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import warnings
warnings.filterwarnings('ignore')


tf.random.set_seed(42)
np.random.seed(42)

import os
os.makedirs('models', exist_ok=True)
os.makedirs('static/images', exist_ok=True)


df = pd.read_csv('data/phisingData.csv')
print(f"Loaded {df.shape[0]} samples with {df.shape[1]} features")

# 2. Prepare Features and Target
X = df.iloc[:, :-1].values  # All columns except last
y = df.iloc[:, -1].values   # Last column (Result)

# Convert -1 to 0 for binary classification (0 = legitimate, 1 = phishing)
y = np.where(y == -1, 0, 1)

print(f"Feature shape: {X.shape}")
print(f"Target distribution: Legitimate={np.sum(y==0)}, Phishing={np.sum(y==1)}")

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training: {X_train.shape[0]} samples")
print(f"Testing: {X_test.shape[0]} samples")

# 4. Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, 'models/scaler.pkl')
print("Scaler saved to models/scaler.pkl")

# 5. Reshape for 1D CNN (samples, timesteps, features)
# CNN expects shape: (batch_size, timesteps, features)
X_train_cnn = np.expand_dims(X_train_scaled,axis=-1)
X_test_cnn = np.expand_dims(X_test_scaled,axis=-1)

print(f"Training CNN Shape:{X_train_cnn.shape}")

# 6. Build CNN Model
model = Sequential([
    # First Conv1D Layer
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.25),
    
    # Second Conv1D Layer
    Conv1D(filters=128, kernel_size=3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.25),
    
    # Third Conv1D Layer
    Conv1D(filters=256, kernel_size=3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.25),
    
    # Flatten and Dense layers
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binary classification
])

# 7. Compile Model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

model.summary()

# 8. Train with Early Stopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

print("\nTraining Model...")
history = model.fit(
    X_train_cnn, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# 9. Save Model
model.save('models/phishing_cnn_model.h5')
print("Model saved to models/phishing_cnn_model.h5")

# 10. Evaluate Model
print("\nEvaluating Model...")
y_pred_proba = model.predict(X_test_cnn)
y_pred = (y_pred_proba > 0.5).astype(int)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


print("MODEL PERFORMANCE")

print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")


# 11. Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))

# 12. Confusion Matrix

print("\Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Legitimate', 'Phishing'],
            yticklabels=['Legitimate', 'Phishing'])
plt.title('Confusion Matrix - Phishing Detection')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# 🔑 IMPORTANT: Save first, then show
plt.savefig('static/images/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print("Confusion matrix saved to static/images/confusion_matrix.png")

# 13. Plot & Save Training History
plt.figure(figsize=(10, 4))

# 1. Left Side Plot: Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()

# 2. Right Side Plot: Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()

plt.tight_layout()

# CRITICAL FIX: Pehle save karein, fir show karein!
plt.savefig('static/images/training_history.png', dpi=300, bbox_inches='tight')
plt.show() # Isko aakhiri mein chalana hai

print("\nTraining complete!")
print("Visualizations saved to static/images/")
