# Age and Gender Prediction using UTKFace Dataset

This project is a deep learning-based age and gender predictor built using the [UTKFace dataset](https://www.kaggle.com/datasets/jangedoo/utkface-new). The model is trained with TensorFlow and Keras and utilizes various data science libraries including Pandas, NumPy, Seaborn, and Matplotlib for preprocessing and visualization.

## 🔍 Overview

The UTKFace dataset contains over 20,000 face images with annotations of age, gender, and ethnicity. This project focuses on:

- Predicting **Age** (as a regression problem)
- Predicting **Gender** (as a binary classification problem)

## 🛠️ Tech Stack

- Python
- Pandas & NumPy
- Matplotlib & Seaborn
- TensorFlow & Keras
- tqdm
- warnings (for cleaner output)

## 📁 Dataset

The [UTKFace dataset](https://www.kaggle.com/datasets/jangedoo/utkface-new) contains cropped and aligned face images with filenames formatted as:

[age][gender][race]_[date&time].jpg


Where:
- `age` is an integer between 0 and 116
- `gender` is 0 (male) or 1 (female)

Example: `25_0_2_20170116174525125.jpg.chip.jpg` → Age: 25, Gender: Male

## 📊 Data Preprocessing

- Parsed labels (age and gender) from image filenames
- Resized all images to a fixed dimension (e.g. 200x200)
- Normalized pixel values
- One-hot encoded gender labels (if needed for training)
- Split data into training and testing sets

## 🧠 Model Architecture

Built using Keras with a Convolutional Neural Network (CNN):

- **Input Layer**: Resized image (e.g. 200x200x3)
- **Conv2D + MaxPooling Layers**: Feature extraction
- **Flatten + Dense Layers**
- **Output Layers**:
  - One node with linear activation for **age regression**
  - One node with sigmoid activation for **gender classification**

## 🏋️ Training

- Loss Functions:
  - Mean Squared Error (MSE) for Age
  - Binary Crossentropy for Gender
- Optimizer: Adam
- Evaluation Metrics: Accuracy (for gender), MAE/MSE (for age)

## 📈 Results

- Visualized training and validation loss/accuracy
- Plotted sample predictions vs. ground truth
- Displayed confusion matrix for gender classification
