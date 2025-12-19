# 🧠 Detecting COVID-19 with Chest X-Ray using PyTorch

This project demonstrates how to build and train a deep learning model to classify chest X-ray images into three categories: **Normal**, **Viral Pneumonia**, and **COVID-19**. It was created as part of the Coursera Guided Project: "Detecting COVID-19 with Chest X-Ray using PyTorch".

## 📊 Project Overview

The goal of this project is to use transfer learning with a pre-trained **ResNet-18** model to automatically detect COVID-19 infection from medical X-ray images.

By leveraging the **COVID-19 Radiography Dataset** from Kaggle, the notebook shows a full deep learning pipeline — from data preparation and augmentation to model training and evaluation — using PyTorch.

## 📁 Dataset

- **Source**: [COVID-19 Radiography Database – Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
- **Classes**:
  - 🟢 Normal
  - 🟡 Viral Pneumonia
  - 🔴 COVID-19

A portion of 30 random images per class is reserved for testing, while the remaining images are used for training.

## ⚙️ Workflow

### 1. Data Preparation
- Reorganizes the dataset into training and testing directories
- Randomly splits data (30 test images per class)
- Displays samples for visualization

### 2. Custom Dataset Loader
- Built using `torch.utils.data.Dataset`
- Applies transformations (resize, normalization, horizontal flip)
- Loads images dynamically during training

### 3. Model Architecture
- **Base Model**: ResNet-18 (pretrained on ImageNet)
- **Modified Layer**: Final fully connected layer → 3 output classes
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam (learning rate = 3e-5)

### 4. Training
- One training epoch with validation after every 20 steps
- Stops early when validation accuracy ≥ 95%
- Displays live validation accuracy and predictions

### 5. Evaluation
- Visualizes true vs predicted labels
- Uses color coding: ✅ Green (Correct) | ❌ Red (Incorrect)

## 📈 Results

- **Validation Accuracy**: ~96.7% after one epoch
- **Loss**: Steadily decreases during training
- The model accurately classifies most X-ray images with strong generalization performance

## 🧰 Technologies Used

- **Language**: Python 3
- **Framework**: PyTorch
- **Libraries**: Torchvision, PIL, NumPy, Matplotlib
- **Model**: ResNet-18 (Transfer Learning)
- **Dataset Source**: Kaggle

## ▶️ How to Run

### 1. Clone the repository:
```bash
git clone https://github.com/NesrineCharrada/CNN-image-anomaly-detection-project.git
cd CNN-image-anomaly-detection-project
```

### 2. Create and activate virtual environment:
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies:
```bash
pip install -r requirements.txt
```

### 4. Download the dataset:
- Visit [Kaggle COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
- Download and extract the dataset
- Place it in the project directory

### 5. Run the Jupyter Notebook:
```bash
jupyter notebook
```

## 📝 Note

This project was completed as part of a Coursera Guided Project. The implementation follows best practices for medical image classification using deep learning and demonstrates the power of transfer learning for healthcare applications.

## 👤 Author

**Nesrine Charrada**
- GitHub: [@NesrineCharrada](https://github.com/NesrineCharrada)

## 📜 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- Coursera Guided Project: "Detecting COVID-19 with Chest X-Ray using PyTorch"
- Kaggle COVID-19 Radiography Database contributors
- PyTorch and torchvision teams