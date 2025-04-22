# 🎵 Audio Genre Classification with DistilHuBERT

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vaxY4J2qUZiVOup_dYmBOuT8qw0MFtij)
[![HuggingFace Model](https://img.shields.io/badge/🤗%20HuggingFace-Model-yellow)](https://huggingface.co/models)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A deep learning project for automatic music genre classification using the GTZAN dataset and fine-tuning a DistilHuBERT audio model.

## 📊 Project Overview

This project demonstrates how to build an effective music genre classifier using transfer learning with the DistilHuBERT model from Hugging Face. The classifier can recognize 10 different music genres from audio samples including rock, jazz, classical, and more.

### Features

- 🔊 Processing audio data with proper normalization and resampling
- 🧠 Fine-tuning a pre-trained DistilHuBERT model for audio classification
- 📈 Performance monitoring and evaluation during training
- 🚀 Model deployment to Hugging Face Hub

## 🗂️ Dataset

The project uses the [GTZAN dataset](https://huggingface.co/datasets/marsyas/gtzan), which contains 1000 audio tracks, each 30 seconds long. The tracks are evenly distributed among 10 genres:
- Blues
- Classical
- Country
- Disco
- Hip-hop
- Jazz
- Metal
- Pop
- Reggae
- Rock

## 🛠️ Installation & Setup

```bash
# Clone this repository
git clone https://github.com/AhemdMahmoud/-GenreWave.git
cd audio-genre-classification

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
transformers
datasets
evaluate
numpy
gradio (optional, for demo)
wandb (optional, for experiment tracking)
```

## 💻 Usage

### Training

```python
# Import and process the GTZAN dataset
from datasets import load_dataset, Audio
gtzan = load_dataset("marsyas/gtzan", "all")
gtzan = gtzan["train"].train_test_split(seed=42, shuffle=True, test_size=0.2)

# Preprocess audio data
gtzan = gtzan.cast_column("audio", Audio(sampling_rate=16000))

# Fine-tune the model
python train.py --epochs 12 --batch_size 8 --learning_rate 5e-5
```

### Inference

```python
from transformers import pipeline

# Load your fine-tuned model
classifier = pipeline("audio-classification", model="Ah7med/distilhubert-finetuned-gtzan")

# Predict genre from audio file
result = classifier("path/to/audio/file.wav")
print(f"Predicted genre: {result[0]['label']}")
```

## 🧠 Model Architecture

The project fine-tunes the [DistilHuBERT model](https://huggingface.co/ntu-spml/distilhubert) from NTU-SPML, which is a distilled version of the HuBERT model. Key aspects of the approach:

1. **Audio Preprocessing**:
   - Resampling to 16kHz to match model requirements
   - Audio normalization (zero mean and unit variance)
   - Handling variable length inputs with truncation

2. **Training Strategy**:
   - Fine-tuning with AdamW optimizer
   - Learning rate: 5e-5 with 10% warmup
   - Mixed precision training (FP16)
   - 12 epochs with early stopping

## 📈 Performance

The model achieves competitive accuracy on the GTZAN test set. Full evaluation metrics are available on the model card on Hugging Face Hub.

### Confusion Matrix

(Add your confusion matrix visualization here)

## 🚀 Future Improvements

- Implement data augmentation (pitch shifting, time stretching)
- Add mel-spectrogram visualization for audio samples
- Explore model quantization for faster inference
- Create a web application for real-time genre classification

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👏 Acknowledgements

- The [GTZAN dataset](https://marsyas.info/index.html) by George Tzanetakis
- Hugging Face for the transformers library and model hosting
- NTU-SPML for the DistilHuBERT model

## 🔗 References

1. Tzanetakis, G., & Cook, P. (2002). Musical genre classification of audio signals. IEEE Transactions on speech and audio processing.
2. Yang, C. et al. (2021). DistilHuBERT: Speech representation learning by layer-wise distillation of hidden-unit BERT.

---

Made with ❤️ by Ahmed Mahmoud
