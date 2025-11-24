# voice-authentication-matlab
A deep-learning based Speaker Recognition system using MFCC features, CNN/LSTM models, and VoxCeleb dataset, with complete pipeline for preprocessing, training, evaluation, and custom dataset handling.

# 🔊 **Speaker Recognition Using Deep Learning**

A deep learning pipeline for **speaker identification** and **speaker verification** using MFCC feature extraction, neural networks (CNN/LSTM), and audio preprocessing on large-scale datasets like **VoxCeleb**.

This project includes:

✔ Audio preprocessing
✔ Feature extraction
✔ Dataset creation
✔ Custom ODataset
✔ Speaker classification model
✔ Training & evaluation pipeline
✔ WAV/Subset datasets
✔ Ready-to-run scripts

---

## 📁 **Folder Structure**

```
📦 Speaker-Recognition-Project/
│
├── dataset_subset/        # Smaller cleaned dataset (processed clips)
├── dataset_wav/           # Raw WAV files
├── models/                # Trained model weights (.pth)
├── ODataset/              # Custom dataset class implementation
├── scripts/               # Preprocessing + training + evaluation code
│
└── README.md
```

---

## 🎯 **Project Objective**

To build an efficient speaker recognition system capable of identifying a speaker’s identity from voice samples using neural network models and audio feature engineering.

## **Working Video**
https://drive.google.com/file/d/1zojf5Du85wSGxKVhdgsEekbxiLEKMRi8/view?usp=sharing 

## 🧠 **Features**

### 🔹 **1. Audio Preprocessing**

* Silence removal
* WAV normalization
* Resampling (16 kHz)
* Segment extraction

### 🔹 **2. Feature Extraction**

* MFCC
* Mel Spectrogram
* Log-Mel Features

### 🔹 **3. Deep Learning Model**

* CNN / LSTM / Hybrid network
* Softmax classification head
* CrossEntropy loss

### 🔹 **4. Dataset Management**

* VoxCeleb integration
* Custom ODataset for training
* Dataset subset support
* Automatic speaker ID mapping

### 🔹 **5. Training Pipeline**

* Batch loading
* Validation set
* Learning rate scheduling

### 🔹 **6. Evaluation**

* Accuracy
* Loss curves
* Confusion matrix
* Prediction on test samples

# 📥 **Dataset Download**

The project supports **VoxCeleb1** dataset.

### **🔗 Official VoxCeleb Dataset Link:**

👉 [https://www.robots.ox.ac.uk/~vgg/data/voxceleb/](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/)

### **Includes:**

* VoxCeleb1
* VoxCeleb2
* Metadata & speaker lists
* Audio files in WAV/M4A

## 🛠️ **Installation & Setup**

### ✔ Clone the repository

```bash
git clone https://github.com/your-username/speaker-recognition.git
cd speaker-recognition
```

### ✔ Create virtual environment

```bash
python -m venv venv
source venv/bin/activate     # Linux/Mac
venv\Scripts\activate        # Windows
```

### ✔ Install dependencies

```bash
pip install -r requirements.txt
```

## 📌 **Usage**

### ✔ **1. Preprocess Dataset**

```bash
python scripts/preprocess_audio.py --dataset dataset_wav --output dataset_subset
```

### ✔ **2. Extract Features**

```bash
python scripts/extract_features.py --input dataset_subset --output features/
```

### ✔ **3. Train Model**

```bash
python scripts/train.py --dataset dataset_subset --epochs 50 --batch 32
```

### ✔ **4. Evaluate Model**

```bash
python scripts/evaluate.py --model models/best_model.pth --dataset dataset_subset
```

### ✔ **5. Test With Custom Audio**

```bash
python scripts/predict.py --audio sample.wav
```

## 📊 **Expected Results**

* 80–95% accuracy depending on dataset size
* Good performance on VoxCeleb subset
* Real-time speaker prediction with optimized model

## 📈 **Future Enhancements**

* Implement **Speaker Verification (Siamese Networks)**
* Add **X-Vectors** or **ECAPA-TDNN embeddings**
* Add Web Interface (Flask/React)
* Deploy on cloud GPU
* Live microphone inference

