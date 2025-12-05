# 🐱🐶 Cat & Dog Classifier

A deep-learning project that classifies images as **Cat** or **Dog** using a Convolutional Neural Network (CNN).
Built with **TensorFlow/Keras**, trained on labelled image datasets, and optimized using data augmentation and model tuning.

---

## 🚀 Features

* CNN model built **from scratch** (with optional MobileNetV2 transfer learning).
* **Data augmentation pipeline** to improve generalization.
* **Training pipeline** with callbacks (EarlyStopping, ModelCheckpoint).
* **Reproducibility support** (random seed locking).
* Evaluation with:

  * Accuracy & loss curves
  * Confusion matrix
  * Classification report
* Script-based flow for:

  * Dataset preparation
  * Model training
  * Model evaluation
  * Inference on new images

---

## 🗂️ Project Structure

```
project/
│
├── train.py                 # Model training script
├── evaluate.py              # Model evaluation script
├── predict.py               # Inference on new images
├── model/                   # Saved models (.h5 or SavedModel)
├── datasets/                # Cat/Dog images (Available at : https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)
│   ├── train/
│   └── test/
└── README.md
```

---

## 🧠 Model Architecture (Scratch)

* Input layer (rescaled)
* Data augmentation
* Conv2D → ReLU → MaxPool
* Conv2D → ReLU → MaxPool
* Flatten
* Dense + Dropout
* Output layer (sigmoid for binary classification)

Loss function:
binary_crossentropy

Optimizer:
Adam

---

## 🔧 Training

Key training components:

* `binary_crossentropy` for two-class output
* `EarlyStopping` to prevent overfitting
* Validation split for monitoring performance

---

## 📊 Evaluation

Generates:

* Accuracy, loss
* Confusion matrix
* Classification report
* Best threshold (if implemented)

---

## 🧬 Reproducibility (optional)

```python
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
```

---

## 📝 Future Improvements

* Hyperparameter tuning
* Better augmentation pipeline
* Transfer learning with MobileNetV2, EfficientNet
* Deployment using Flask/FastAPI
