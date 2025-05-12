# 🚗 State Farm Driver Distraction Detection

This project classifies driver behaviors using deep learning models trained on the [State Farm Distracted Driver Detection dataset](https://www.kaggle.com/c/state-farm-distracted-driver-detection). It leverages transfer learning with ResNet50 and DenseNet121 to accurately predict the driver's activity, such as texting, talking, or safe driving.

## 📌 Project Overview

- **Objective**: Detect and classify 10 categories of driver behavior from dashboard camera images.
- **Dataset**: Provided by State Farm, includes over 22,000 labeled images.
- **Models Used**:
  - **ResNet50** (Transfer Learning)
  - **DenseNet121** (Transfer Learning)
- **Frameworks**: TensorFlow, Keras, OpenCV, Matplotlib
- **Accuracy Achieved**: Up to **92.85%** on validation data.

## 📁 Project Structure

```
.
├── state-farm.ipynb       # Main notebook containing the entire pipeline
├── README.md              # Project documentation (you are here)
└── model/                 # Folder to save trained models (optional)
```

## 🔍 Driver Classes

The dataset includes the following 10 classes:

- `c0`: Safe driving  
- `c1`: Texting – right hand  
- `c2`: Talking on the phone – right hand  
- `c3`: Texting – left hand  
- `c4`: Talking on the phone – left hand  
- `c5`: Operating the radio  
- `c6`: Drinking  
- `c7`: Reaching behind  
- `c8`: Hair and makeup  
- `c9`: Talking to passenger

## 🧠 Model Pipeline

1. **Data Preprocessing**:
   - Image resizing and normalization
   - Augmentation using ImageDataGenerator
2. **Model Training**:
   - Feature extraction using pretrained CNNs
   - Fine-tuning on classification head
   - Early stopping and checkpointing
3. **Evaluation**:
   - Accuracy, Confusion Matrix
   - Visualization of predictions

## 🚀 Results

| Model        | Validation Accuracy |
|--------------|---------------------|
| ResNet50     | 91.52%              |
| DenseNet121  | **92.85%**          |

## 📦 Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
```
