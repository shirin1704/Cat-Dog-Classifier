import os
import random
import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf # pyright: ignore[reportMissingImports]
from tensorflow.keras import layers, models, callbacks, optimizers # pyright: ignore[reportMissingImports]
from tensorflow.keras.preprocessing.image import ImageDataGenerator # pyright: ignore[reportMissingImports]
from tensorflow.keras.models import load_model # pyright: ignore[reportMissingImports]
from tensorflow.keras.preprocessing import image # pyright: ignore[reportMissingImports]
from tensorflow.keras.preprocessing import image_dataset_from_directory # pyright: ignore[reportMissingImports]


#introducing reproducibility
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


train_dir = "Sample_Train"
test_dir = "Sample_Test"

IMG_SIZE = (128, 128)    
BATCH_SIZE = 16
EPOCHS = 60              
SEED = 42

MODEL_SAVE_PATH = "cat_dog_cnn_from_scratch.keras"


train_ds = image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='int',     # produces 0 and 1 labels
    class_names=['cat', 'dog'],  
    image_size=(128, 128),
    batch_size=16,
    shuffle=True
)

test_ds = image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='int',
    class_names=['cat', 'dog'],
    image_size=(128, 128),
    batch_size=16
)

#print(train_ds.class_names)
#print(test_ds.class_names)


AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

#Augmentation helps to generalise and avoid overfitting
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])


model = models.Sequential([
    layers.Input(shape=IMG_SIZE + (3,)),
    layers.Rescaling(1./255),
    data_augmentation,

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.summary()


early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)


history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=20,
    callbacks=[early_stop]
)


loss, acc = model.evaluate(test_ds)
print(f"\n✅ Test Accuracy: {acc:.4f} | Test Loss: {loss:.4f}")


import matplotlib.pyplot as plt

for images, labels in test_ds.take(1):
    preds = model.predict(images)
    preds = (preds > 0.5).astype("int32").flatten()

    plt.figure(figsize=(10,10))
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        # Use the class names from the original dataset before caching/prefetching
        true_label = ['cat', 'dog'][labels[i]]
        pred_label = ['cat', 'dog'][preds[i]]
        color = "green" if pred_label == true_label else "red"
        plt.title(f"T:{true_label}\nP:{pred_label}", color=color)
        plt.axis("off")
    plt.show()
    break

plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.show()


model.save(MODEL_SAVE_PATH)
print(f"\n✅ Model saved to: {MODEL_SAVE_PATH}")

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

y_true, y_pred = [], []
for images, labels in test_ds:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend((preds > 0.5).astype("int32").flatten())

print(classification_report(y_true, y_pred, target_names=['Cat','Dog']))
print(confusion_matrix(y_true, y_pred))