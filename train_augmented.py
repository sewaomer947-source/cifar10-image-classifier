import time
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
print("Loading CIFAR-10 dataset...")
# 1. Load and normalize dataset
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

print("Building Augmented CNN model...")
# 2. Build CNN model with Data Augmentation layers
model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),

    # Data Augmentation layers
    layers.RandomFlip("horizontal"),
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1),

    # Convolution & Pooling Layers
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Starting training for 10 epochs with augmentation...")
# 3. Track training time and train for 10 epochs
start_time = time.time()
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
end_time = time.time()
training_time = end_time - start_time

print(f"Training finished in {training_time:.2f} seconds.")

# 4. Save metrics, training time, and model
with open('training_augmented_metrics.txt', 'w') as f:
    f.write(f"Total Training Time (seconds): {training_time:.2f}\n")
    f.write("Epoch,Loss,Accuracy,Val_Loss,Val_Accuracy\n")
    for i in range(10):
        f.write(
            f"{i + 1},{history.history['loss'][i]:.4f},{history.history['accuracy'][i]:.4f},{history.history['val_loss'][i]:.4f},{history.history['val_accuracy'][i]:.4f}\n")

model.save('cifar10_augmented_model.keras')
print("Augmented model training complete and saved successfully!")