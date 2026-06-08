# AgriVision-AI-Crop-Disease-Detection1
AgriVision AI leverages Deep Learning and Computer Vision to provide fast, accurate, and scalable crop disease diagnosis for smart farming and sustainable agriculture.
# AgriVision Crop Disease Detection

## Problem Statement
Plant diseases significantly reduce crop yield and farmer income when not detected early. Manual inspection is time-consuming, subjective, and not scalable. This project addresses the need for an automated, accurate, and fast crop disease detection system using deep learning and computer vision.

## Overview
This project uses a Convolutional Neural Network (CNN) trained on leaf images to classify healthy and diseased crops, enabling automated and reliable disease diagnosis for smart farming applications.

## Objectives
- Detect diseased crops
- Improve agricultural productivity
- Enable early disease identification

## Tech Stack
- Python
- TensorFlow
- OpenCV
- NumPy
- Matplotlib
- Google Colab

## Dataset
The PlantVillage Dataset is a publicly available, large-scale image dataset containing labeled images of healthy and diseased plant leaves across multiple crop species. The dataset was sourced from Kaggle and used for training, validation, and testing of the deep learning model.

## Week 1
- Dataset Collection
- Image EDA
- Class Distribution Analysis
- Image Preprocessing
- Image Resizing
- Normalization
- Train Validation Test Split
- Data Augmentation

## Week 2
- Custom CNN Model
- Conv2D Layers
- MaxPooling Layers
- Dropout
- EarlyStopping
- Model Training

# Model Performance

- Training Accuracy: 82%
- Validation Accuracy: 87%
- Test Accuracy: 86.43%
- Test Loss: 0.3834

# Week 3
- Loaded PlantVillage dataset
- Resized images for model input
- Normalized image pixel values
- Applied data augmentation techniques
- Used CNN with MobileNetV2 architecture
- Implemented transfer learning with ImageNet weights
- Added custom classification layers
- Trained the model on training data
- Evaluated model using validation data

# Week 4
- Created Streamlit web application
- Integrated trained CNN model into the app
- Implemented image upload functionality
- Performed image preprocessing for prediction
- Displayed predicted crop disease result
- Tested application on localhost
- Configured ngrok for public access
- Generated public URL for live demo
