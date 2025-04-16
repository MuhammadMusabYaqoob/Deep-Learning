# 🧠 MNIST Handwritten Digit Classification using Deep Learning

This project demonstrates how to train and test a deep learning model on the **MNIST handwritten digit dataset** using **TensorFlow** and **Keras**. It includes both the model training and real-world testing using custom images, ideal for showcasing digit recognition capabilities.

---

## 📁 Files Overview

- `MNIST Handwritten Digit Classification using Deep Learning (Neural Network).ipynb`  
  → Model development, training, and evaluation on MNIST dataset

- `predictions.ipynb`  
  → Testing the trained model on a **custom image** (captured externally, e.g., from a mobile camera)

---

## 🛠️ Project Structure & Key Steps

### 1. **Data Loading & Visualization**
- Loaded MNIST dataset using `tf.keras.datasets.mnist.load_data()`
- Visualized the first 10 digits with `matplotlib.pyplot`
- Observed grayscale 28x28 pixel format
- Dataset shape:
  - Training: 60,000 samples
  - Testing: 10,000 samples

### 2. **Preprocessing**
- Normalized pixel values (0–255 scaled to 0–1)
- Flattened 28x28 images into 1D vectors (784 features) for dense layers

### 3. **Model Architecture**
A Sequential deep neural network:
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```
- **Optimizer**: Adam  
- **Loss Function**: Sparse Categorical Crossentropy  
- **Metrics**: Accuracy

### 4. **Model Training**
- Trained for 10 epochs
- Achieved high accuracy on the validation set

### 5. **Model Evaluation**
- Evaluated on test dataset
- Printed accuracy score
- Displayed predictions with actual labels using `matplotlib`

---

## 📷 Real-World Testing with Custom Image

### File: `predictions.ipynb`

Tested the trained model with a new digit image:

#### 1. **Image Input**
- Loaded a photo (`MNIST_digit.png`) using `cv2.imread()`
- Converted to grayscale
- Resized to 28x28 pixels for model compatibility

#### 2. **Preprocessing**
- Inverted pixel values (since MNIST digits are white on black)
- Normalized and reshaped for prediction

#### 3. **Prediction**
- Used the trained model to predict the digit
- Visualized input and displayed predicted class

---

## 🧪 Results

- ✅ Model performs with high confidence on standard MNIST test data
- ✅ Successfully predicts handwritten digits from external sources

---

## 📦 Dependencies

Ensure the following Python libraries are installed:

```bash
tensorflow
numpy
matplotlib
opencv-python
```

---

## 💾 Model Saving & Loading

- The trained model is saved using:
  ```python
  model.save("models_saved/model.keras")
  ```
- Later loaded for testing using:
  ```python
  tf.keras.models.load_model("models_saved/model.keras")
  ```

---

## 📌 Notes & Observations

- Good preprocessing (grayscale, resizing, normalization) is **crucial** when testing with real images.
- Model generalizes well to new data if trained properly.

---

## 📤 Project Deployment

Ideal for:
- Digit recognition demos
- AI education projects
- Beginner-friendly deep learning experiments
