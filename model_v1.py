# cnn_from_scratch.py
import os
import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf # pyright: ignore[reportMissingImports]
from tensorflow.keras import layers, models, callbacks, optimizers # pyright: ignore[reportMissingImports]
from tensorflow.keras.preprocessing.image import ImageDataGenerator # pyright: ignore[reportMissingImports]
from tensorflow.keras.models import load_model # pyright: ignore[reportMissingImports]

# --------------------
# CONFIG
# --------------------
train_dir = "Sample_Train"
test_dir = "Sample_Test"

IMG_SIZE = (128, 128)     # width, height
BATCH_SIZE = 16
EPOCHS = 60               # we'll use EarlyStopping
SEED = 42

MODEL_SAVE_PATH = "cat_dog_cnn_from_scratch.h5"

# --------------------
# DATA GENERATORS
# --------------------
# Augmentation for training (helps when dataset is small)
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.12,
    height_shift_range=0.12,
    shear_range=0.08,
    zoom_range=0.12,
    horizontal_flip=True,
    fill_mode='nearest'
)

# For validation/test we only rescale
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',   # cat/dog -> binary
    seed=SEED,
    shuffle=True
)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Print class indices
print("Class indices:", train_gen.class_indices)

# --------------------
# BUILD THE MODEL
# --------------------
def build_cnn(input_shape=(128,128,3)):
    inputs = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(32, (3,3), padding='same', activation=None)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2,2))(x)

    # Block 2
    x = layers.Conv2D(64, (3,3), padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.2)(x)

    # Block 3
    x = layers.Conv2D(128, (3,3), padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, (3,3), padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.3)(x)

    # Block 4 (smaller filters)
    x = layers.SeparableConv2D(256, (3,3), padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.35)(x)

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)   # binary

    model = models.Model(inputs, outputs, name="CatDog_CNN_from_scratch")
    return model

model = build_cnn(input_shape=(IMG_SIZE[1], IMG_SIZE[0], 3))
model.summary()

# --------------------
# COMPILE
# --------------------
optimizer = optimizers.Adam(learning_rate=1e-4)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

# --------------------
# CALLBACKS
# --------------------
checkpoint_cb = callbacks.ModelCheckpoint(
    MODEL_SAVE_PATH, monitor='val_auc', mode='max', save_best_only=True, verbose=1
)
earlystop_cb = callbacks.EarlyStopping(
    monitor='val_auc', mode='max', patience=10, restore_best_weights=True, verbose=1
)
reduce_lr_cb = callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=1
)

# --------------------
# TRAIN
# --------------------
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=test_gen,
    callbacks=[checkpoint_cb, earlystop_cb, reduce_lr_cb],
    verbose=2
)

# --------------------
# PLOT TRAINING CURVES
# --------------------
def plot_history(hist):
    h = hist.history
    epochs_range = range(1, len(h['loss']) + 1)

    plt.figure(figsize=(12,4))

    plt.subplot(1,2,1)
    plt.plot(epochs_range, h['loss'], label='train_loss')
    plt.plot(epochs_range, h['val_loss'], label='val_loss')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs_range, h['accuracy'], label='train_acc')
    plt.plot(epochs_range, h['val_accuracy'], label='val_acc')
    plt.title('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_history(history)

# --------------------
# EVALUATE ON TEST
# --------------------
print("\nEvaluation on test set (generator):")
eval_results = model.evaluate(test_gen, verbose=2)
for name, val in zip(model.metrics_names, eval_results):
    print(f"{name}: {val:.4f}")

# --------------------
# SAVE (already saved best) AND HOW TO LOAD
# --------------------
print(f"\nBest model saved to {MODEL_SAVE_PATH}")
# To load later: model = load_model(MODEL_SAVE_PATH)

# --------------------
# PREDICTIONS ON A FEW TEST IMAGES
# --------------------
import numpy as np
from tensorflow.keras.preprocessing import image # pyright: ignore[reportMissingImports]

def predict_from_path(img_path, model, target_size=IMG_SIZE):
    img = image.load_img(img_path, target_size=target_size)
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    pred = model.predict(arr)[0][0]
    return pred

# Show a few images and preds
test_paths = []
for root, _, files in os.walk(test_dir):
    for f in files:
        if f.lower().endswith(('.jpg','.jpeg','.png')):
            test_paths.append(os.path.join(root,f))
# show up to 8 random
sample_paths = np.random.choice(test_paths, size=min(8, len(test_paths)), replace=False)
plt.figure(figsize=(12,6))
for i, p in enumerate(sample_paths):
    pred = predict_from_path(p, model)
    label = "dog" if pred >= 0.5 else "cat"
    prob = pred if pred>=0.5 else 1.0-pred
    plt.subplot(2,4,i+1)
    img = image.load_img(p, target_size=IMG_SIZE)
    plt.imshow(img)
    plt.title(f"{label} ({prob:.2f})")
    plt.axis('off')
plt.tight_layout()
plt.show()