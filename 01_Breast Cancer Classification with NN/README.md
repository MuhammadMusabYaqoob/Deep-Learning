
# 🧠 Breast Cancer Classification using a Neural Network

## 🔍 Project Summary
The objective of this project is to build a neural network that can classify whether a breast cancer tumor is **malignant** or **benign** using features derived from digitized images.

---

## 📊 Dataset
- **Source**: Built-in dataset from `sklearn.datasets`
- **Features**: 30 numeric features (mean radius, texture, perimeter, etc.)
- **Target**: Binary label — 0 (malignant), 1 (benign)

---

## 📁 Step-by-Step Workflow

### 1. Importing Libraries
Imported essential libraries including `NumPy`, `Pandas`, `Matplotlib`, `Seaborn`, `Sklearn`, and `TensorFlow/Keras`.

### 2. Data Exploration
- Loaded the dataset using `sklearn.datasets.load_breast_cancer()`
- Converted to DataFrame and explored with `.head()`, `.info()`, etc.
- Visualized correlations using heatmaps and histograms

### 3. Preprocessing
- Split into train-test sets using `train_test_split()`
- Standardized features with `StandardScaler`

### 4. Neural Network Model
- Built a simple Sequential model with:
  - Two Dense layers (64 units, ReLU)
  - One Output layer (1 unit, Sigmoid)
- Compiled using `adam` and `binary_crossentropy`
- Trained over 100 epochs with batch size of 32

### 5. Evaluation
- Achieved high test accuracy
- Used confusion matrix and classification report for evaluation

---

## ✅ Results
- **High training and testing accuracy**
- **Well-balanced performance with no overfitting**
- **Effective in binary classification task**

---

## 🚀 Future Work
- Add dropout layers for regularization
- Perform hyperparameter tuning
- Try alternative architectures and optimizers

---

## 📂 Files
- `Breast Cancer Classification with NN.ipynb` — Main notebook

---

## ✍️ Author
- Muhammad Musab
