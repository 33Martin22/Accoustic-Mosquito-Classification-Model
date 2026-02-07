# Accoustic-Mosquito-Classification-Model
# Acoustic Mosquito Species Classification using Deep Learning

A deep learning project for classifying mosquito species based on their acoustic signatures using multiple neural network architectures with Bayesian uncertainty estimation.

## 🦟 Overview

This project implements multiple deep learning models to classify four mosquito species using audio recordings of their wingbeat frequencies. The models employ Bayesian Neural Networks (BNN) with Monte Carlo Dropout for uncertainty quantification, making the predictions more reliable and interpretable.

### Target Species
- **Ae_aegypti** (Aedes aegypti) - Yellow fever mosquito
- **Ae_albopictus** (Aedes albopictus) - Asian tiger mosquito  
- **An_gambiae** (Anopheles gambiae) - Malaria mosquito
- **C_quinquefasciatus** (Culex quinquefasciatus) - Southern house mosquito

## 🎯 Key Features

- **Three Model Architectures**:
  - Custom MozzBNNv2 with CBAM attention mechanism
  - Modified ResNet18 with Bayesian layers
  - VGGish pretrained model with custom classifier
  
- **Bayesian Uncertainty Quantification**: Monte Carlo Dropout for prediction confidence
- **Audio Feature Extraction**: Log-mel spectrograms from raw audio
- **Class Imbalance Handling**: Weighted sampling and data augmentation
- **Comprehensive Evaluation**: Confusion matrices, ROC curves, precision-recall curves
- **Interactive Inference**: Gradio interfaces for real-time predictions

## 📋 Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architectures](#model-architectures)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results Visualization](#results-visualization)
- [Gradio Interface](#gradio-interface)
- [Requirements](#requirements)
- [Project Structure](#project-structure)

## 🔧 Installation

### Prerequisites
- Python 3.7+
- CUDA-capable GPU (recommended)
- Google Colab (optional, but notebook is optimized for it)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/33Martin22/Acoustic-mosquito-classification.git
cd acoustic-mosquito-classification
```

2. Install required packages:
```bash
pip install torch torchvision torchaudio
pip install librosa soundfile
pip install pandas numpy matplotlib seaborn
pip install scikit-learn
pip install gradio
pip install torchvggish
```

3. For Google Colab users:
```python
from google.colab import drive
drive.mount('/content/drive')
```

## 📊 Dataset

### Dataset Structure
The project expects:
- **Audio files**: `.wav` format mosquito recordings
- **Metadata CSV**: Contains species labels and file paths

### Data Preprocessing
1. **Audio Windowing**: Each audio file is divided into 20ms windows
2. **Feature Extraction**: Log-mel spectrograms (64 mel bands)
3. **Normalization**: Min-max scaling to [0, 1] range
4. **Class Balancing**: Weighted sampling to handle imbalanced classes

### Expected Metadata Format
```csv
species,file_path
Ae_aegypti,audio/file1.wav
C_quinquefasciatus,audio/file2.wav
...
```

## 🏗️ Model Architectures

### 1. MozzBNNv2 (Custom CNN with CBAM)

A custom Convolutional Neural Network with:
- 4 convolutional blocks with BatchNorm
- CBAM (Convolutional Block Attention Module) for feature refinement
- Monte Carlo Dropout for Bayesian inference
- Input: (1, 64, 96) spectrograms

**Architecture Highlights**:
```
Conv2D(32) → BatchNorm → CBAM → MaxPool → Dropout
Conv2D(64) → BatchNorm → CBAM → MaxPool → Dropout
Conv2D(128) → BatchNorm → CBAM → MaxPool → Dropout
Conv2D(256) → BatchNorm → CBAM → MaxPool → Dropout
Flatten → FC(128) → Dropout → FC(4)
```

### 2. Modified ResNet18 with BNN

Transfer learning approach using ResNet18:
- Modified first layer for single-channel input
- Bayesian BasicBlocks with dropout
- Pretrained on ImageNet (optional)
- Fine-tuned for mosquito classification

### 3. VGGish Pretrained Model

Audio-specific pretrained model:
- VGGish embeddings (frozen)
- Custom classifier head with dropout
- Optimized for audio classification tasks

## 💻 Usage

### Quick Start

```python
import torch
import librosa
import numpy as np

# Load the trained model
model = mozzBNNv2(input_shape=(1, 64, 96), num_classes=4)
model.load_state_dict(torch.load("model_a.pth"))
model.eval()

# Load and preprocess audio
audio, sr = librosa.load("mosquito_sample.wav", sr=16000)
mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
mel_db = librosa.power_to_db(mel, ref=np.max)

# Normalize
mel_normalized = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())

# Predict
input_tensor = torch.FloatTensor(mel_normalized).unsqueeze(0).unsqueeze(0)
with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.argmax(output, dim=1)

label_map = {0: "Ae_aegypti", 1: "Ae_albopictus", 
             2: "An_gambiae", 3: "C_quinquefasciatus"}
print(f"Predicted species: {label_map[prediction.item()]}")
```

## 🎓 Training

### Training MozzBNNv2

```python
# Initialize model
model = mozzBNNv2(input_shape=(1, 64, 96), num_classes=4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training configuration
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 25

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Validation
    model.eval()
    val_accuracy = evaluate_model(model, val_loader, device)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Val Acc: {val_accuracy:.4f}")
```

### Training ResNet18

```python
model_B = modify_resnet18_for_bnn(num_classes=4, pretrained=True)
train_resnet_model(model_B, train_loader, val_loader, epochs=18, lr=0.001)
```

### Hyperparameters

| Parameter | MozzBNNv2 | ResNet18 | VGGish |
|-----------|-----------|----------|--------|
| Learning Rate | 0.001 | 0.001 | 0.0001 |
| Epochs | 25 | 18 | 15 |
| Batch Size | 64 | 64 | 32 |
| Dropout | 0.3 | 0.3 | 0.5 |
| Optimizer | Adam | Adam | Adam |

## 📈 Evaluation

### Monte Carlo Dropout Evaluation

```python
def evaluate_mc_dropout(model, val_loader, mc_runs=30):
    model.train()  # Keep dropout active
    correct, total = 0, 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Multiple forward passes
            predictions = []
            for _ in range(mc_runs):
                outputs = model(inputs)
                predictions.append(outputs)
            
            # Average predictions
            avg_prediction = torch.stack(predictions).mean(dim=0)
            predicted = torch.argmax(avg_prediction, dim=1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total
```

### Performance Metrics

The notebook includes functions for:
- **Confusion Matrix**: Visual representation of classification performance
- **ROC Curves**: One-vs-rest ROC curves for each class
- **Precision-Recall Curves**: Per-class precision-recall analysis
- **t-SNE Visualization**: Feature space visualization

## 📊 Results Visualization

### Confusion Matrix
```python
plot_confusion_matrix(model, val_loader, device, 
                     class_labels=["Ae_aegypti", "Ae_albopictus", 
                                  "An_gambiae", "C_quinquefasciatus"])
```

### Multi-class Metrics
```python
plot_multiclass_metrics(model, val_loader, device,
                       class_labels=["Ae_aegypti", "Ae_albopictus",
                                    "An_gambiae", "C_quinquefasciatus"],
                       model_name="MozzBNNv2")
```

## 🎨 Gradio Interface

The project includes Gradio interfaces for interactive inference:

```python
import gradio as gr

def predict_mosquito(audio_file):
    features = extract_features(audio_file)
    prediction, confidence = model_predict(features)
    return f"Species: {prediction}, Confidence: {confidence:.2%}"

interface = gr.Interface(
    fn=predict_mosquito,
    inputs=gr.Audio(type="filepath"),
    outputs="text",
    title="Mosquito Species Classifier",
    description="Upload a mosquito audio recording for species classification"
)

interface.launch()
```

## 📦 Requirements

Create a `requirements.txt` file:

```
torch>=1.9.0
torchvision>=0.10.0
torchaudio>=0.9.0
librosa>=0.9.0
soundfile>=0.10.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
gradio>=3.0.0
torchvggish>=0.1.0
```

## 📁 Project Structure

```
acoustic-mosquito-classification/
│
├── Acoustic_MosquitoMSC.ipynb    # Main notebook
├── README.md                      # This file
├── requirements.txt               # Python dependencies
│
├── models/
│   ├── model_a.pth               # MozzBNNv2 weights
│   ├── model_b.pth               # ResNet18 weights
│   └── model_c.pth               # VGGish weights
│
├── data/
│   ├── kagglesounddataset.csv    # Metadata
│   └── all_audio/                # Audio files
│
└── results/
    ├── confusion_matrices/
    ├── roc_curves/
    └── evaluation_metrics/
```

## 🚀 Getting Started

1. **Prepare your data**: Organize audio files and create metadata CSV
2. **Run preprocessing**: Execute data loading and feature extraction cells
3. **Train models**: Choose your model architecture and train
4. **Evaluate**: Use provided evaluation functions
5. **Deploy**: Use Gradio interface for inference

## 🧪 Key Components

### Feature Extraction
- **Sampling Rate**: 16 kHz
- **Window Size**: 20ms
- **Mel Bands**: 64
- **Spectrogram Type**: Log-mel

### Data Augmentation
- Time stretching
- Pitch shifting
- Background noise addition
- SpecAugment (frequency masking)

### Class Imbalance Handling
- Weighted random sampling
- SMOTE (optional)
- Class weights in loss function

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{acoustic_mosquito_classification,
  author = {Your Name},
  title = {Acoustic Mosquito Species Classification using Deep Learning},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/33Martin22/acoustic-mosquito-classification}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- VGGish pretrained model from Google Research
- ResNet architecture from torchvision
- Audio processing libraries: librosa
- Deep learning framework: PyTorch

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact [kiokomartin27@gmail.com]

## 🔗 References

- [VGGish: Audio Event Detection](https://github.com/tensorflow/models/tree/master/research/audioset/vggish)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521)
- [Monte Carlo Dropout for Uncertainty Estimation](https://arxiv.org/abs/1506.02142)

---

⭐ If you find this project useful, please consider giving it a star!
