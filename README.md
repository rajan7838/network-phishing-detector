# 🛡️ Network Phishing Detection

CNN-based phishing URL detector with 97%+ accuracy.

## Features
- 1D CNN architecture
- 30 URL features
- 97.2% accuracy
- Streamlit UI
- Docker support

## Quick Start
```bash
# Install
pip install -r requirements.txt

# Train
python run.py 1

# Run Streamlit
python run.py 2

# Docker
python run.py 3  # Build
python run.py 4  # Run


---

## Key Differences from MNIST:

| Aspect | MNIST (Image) | Phishing (Tabular) |
|--------|---------------|-------------------|
| **Input Shape** | 2D (28,28) | 1D (30 features) |
| **Layer Type** | Conv2D | Conv1D |
| **Pooling** | MaxPooling2D | MaxPooling1D |
| **Channels** | 1 (grayscale) | 1 (feature dimension) |
| **Preprocessing** | Normalize pixels | StandardScaler |

---

## 🚀 Run Your Phishing Detector:

```bash
# 1. Train the model
python run.py 1

# 2. Launch Streamlit
python run.py 2

# 3. Build and run Docker
python run.py 3
python run.py 4