import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import joblib
import os

# Create project folder if not exists (redundant but safe)
os.makedirs("vlsi-ml-accelerator", exist_ok=True)

# Load and preprocess data
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a small MLP (2 hidden layers, very fast)
# Use alpha for regularization to keep weights small
model = MLPClassifier(hidden_layer_sizes=(8, 4), activation='relu', max_iter=2000, alpha=0.1, random_state=42)
model.fit(X_train, y_train)

# Evaluate software accuracy
print(f"Software Model Accuracy: {model.score(X_test, y_test):.4f}")

# Extract weights & biases
weights = [layer.T for layer in model.coefs_] 
biases = model.intercepts_

# Simple 8-bit fixed-point quantization (scale to -128 to 127)
def quantize(arr, scale=16):
    return np.clip(np.round(arr * scale).astype(np.int8), -128, 127)

quant_weights = [quantize(w) for w in weights]
quant_biases = [quantize(b) for b in biases]

# Save everything
np.savez("quantized_model.npz", 
         w1=quant_weights[0], b1=quant_biases[0], 
         w2=quant_weights[1], b2=quant_biases[1], 
         w3=quant_weights[2], b3=quant_biases[2], 
         scaler_mean=scaler.mean_, 
         scaler_scale=scaler.scale_)

print("Quantized model saved! Ready for hardware.")
joblib.dump(scaler, "scaler.pkl")
# Save software reference for comparison
joblib.dump((model, scaler, X_test, y_test), "software_model.pkl")
