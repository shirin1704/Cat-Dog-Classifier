import tensorflow as tf # pyright: ignore[reportMissingImports]
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # pyright: ignore[reportMissingImports]
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import random

# -----------------------------
# 1️⃣ Load saved model
# -----------------------------
model_path = "cat_dog_cnn_from_scratch_works1.keras"  # Update path if needed
model = tf.keras.models.load_model(model_path)
print("✅ Model loaded successfully!")

#model.summary()

# -----------------------------
# 2️⃣ Load test dataset
# -----------------------------
test_dir = "Sample_Test"  # Folder containing Cat/ and Dog/

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(128, 128),   # must match training size
    batch_size=32,
    label_mode='binary',
    shuffle=False
)

class_names = test_ds.class_names  # ['Cat', 'Dog']
#print(class_names)

loss, acc = model.evaluate(test_ds)
print(f"\n✅ Test Accuracy: {acc:.4f} | Test Loss: {loss:.4f}")

# -----------------------------
# 3️⃣ Run predictions
# -----------------------------
y_true = np.concatenate([y for x, y in test_ds], axis=0)
y_true = np.array(y_true).astype(int).flatten()
y_pred_prob = model.predict(test_ds)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

print("Y_True: ", y_true[:10])  # print first 10 true labels
print("Y_Pred: ", y_pred[:10])  # print first 10 predicted classes

# -----------------------------
# 4️⃣ Compute accuracy details
# -----------------------------

correct = np.sum(y_true == y_pred)
incorrect = np.sum(y_true != y_pred)
total = len(y_true)
accuracy = correct / total

print(f"\n🔍 Evaluation Summary")
print(f"Total samples     : {total}")
print(f"✅ Correct preds   : {correct}")
print(f"❌ Incorrect preds : {incorrect}")
print(f"🎯 Accuracy        : {accuracy:.4f}")

# -----------------------------
# 5️⃣ (Optional) Few sample predictions
# -----------------------------
images = []
labels = []
for batch_images, batch_labels in test_ds.unbatch().take(500):  # limit to 500 if test_ds is large
    images.append(batch_images.numpy())
    labels.append(batch_labels.numpy())

images = np.array(images)
labels = np.array(labels).astype(int)

# Get predicted probabilities for all test images
pred_probs = model.predict(images)
pred_classes = (pred_probs > 0.5).astype(int).flatten()

# Choose 15 random indices
sample_indices = random.sample(range(len(images)), 15)

# Plot them
plt.figure(figsize=(15, 10))
for i, idx in enumerate(sample_indices):
    ax = plt.subplot(3, 5, i + 1)
    plt.imshow(images[idx] / 255.0)  # images are already normalized if preprocess_input wasn’t used
    true_label = "Cat" if labels[idx] == 0 else "Dog"
    pred_label = "Cat" if pred_classes[idx] == 0 else "Dog"
    color = "green" if true_label == pred_label else "red"
    plt.title(f"Pred: {pred_label}\nTrue: {true_label}", color=color, fontsize=10)
    plt.axis("off")

plt.tight_layout()
plt.show()

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['Cat', 'Dog']))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))