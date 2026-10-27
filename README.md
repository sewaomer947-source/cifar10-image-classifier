# CIFAR-10 Image Classification Pipeline

A deep learning computer vision project built to train, evaluate, and test convolutional neural network (CNN) architectures using the CIFAR-10 dataset in TensorFlow.

## 🧠 Project Overview
This repository implements an end-to-end computer vision workflow using TensorFlow/Keras to train a multi-class image classifier, evaluate baseline performance on standardized test sets, and test model generalization against unconstrained real-world photographs.

## 🛠️ My Specific Contributions & Engineering Decisions
* **Model Training & Experimentation:** Trained a baseline CNN for 10 epochs using the Adam optimizer and sparse categorical cross-entropy loss with integer labels. Incorporated data augmentation (random horizontal flips and height/width shifts) to compare training convergence against the baseline.
* **Dataset Management & Pipelines:** Built data loading pipelines, standardized 32x32 image normalization, and created an evaluation script for custom image inference.
* **AI-Augmented Workflow:** Utilized AI tools to debug TensorFlow tensor shape mismatches and optimize training loops, while personally directing model hyperparameter choices and evaluation metrics.

## 📊 Concrete Experimental Results
* **Baseline Test Accuracy:** Reached **72%** test accuracy on the clean, 32x32 CIFAR-10 test set.
* **Real-Photo Generalization:** Achieved **25%** accuracy when evaluated on **12** unconstrained real-world photographs taken outside the training distribution.
* **Engineering Insight:** The significant performance drop on custom photos highlighted domain-gap challenges caused by high-resolution noise, complex background clutter, and lighting variations, emphasizing the necessity of domain adaptation and robust data augmentation in computer vision models.

## 🚀 Technologies Used
* Python 3
* TensorFlow / Keras & OpenCV
* NumPy & Matplotlib
* Git & GitHub for Version Control
