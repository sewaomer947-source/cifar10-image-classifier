import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# 1. Define the 10 CIFAR-10 class labels in order
class_names = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck',
]

# 2. Load your trained model
model_path = 'cifar10_augmented_model.keras'
print(f'Loading model from {model_path}...')
model = tf.keras.models.load_model(model_path)

# 3. Path to your folder containing multiple test images
folder_path = 'test_images'

if not os.path.exists(folder_path):
  os.makedirs(folder_path)
  print(f"Created '{folder_path}' folder. Please put your test images inside it!")
else:
  # Find all image files in the folder
  valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
  image_files = [
      f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)
  ]

  if not image_files:
    print(
        f"The '{folder_path}' folder is empty. Drop some images into it and"
        ' run again!'
    )
  else:
    print(f'Found {len(image_files)} image(s) to test. Running predictions...\n')

    # Loop through every image in the folder
    for img_file in image_files:
      img_path = os.path.join(folder_path, img_file)
      try:
        # Load and resize image to 32x32
        img = image.load_img(img_path, target_size=(32, 32))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        probs = tf.nn.softmax(predictions[0]).numpy()
        predicted_class_index = np.argmax(probs)
        confidence = np.max(probs) * 100
        predicted_class_name = class_names[predicted_class_index]

        print('----------------------------------------')
        print(f'Image File: {img_file}')
        print(f'Prediction Result: {predicted_class_name.upper()}')
        print(f'Confidence: {confidence:.2f}%')
        print('----------------------------------------')

      except Exception as e:
        print(f'Could not process {img_file}. Error: {e}')