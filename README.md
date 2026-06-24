# Image Classification Using CNN

## Overview

This project is an Image Classification system built using a Convolutional Neural Network (CNN) and the CIFAR-10 dataset. The model can classify images into 10 different categories and provides prediction confidence scores through a user-friendly Streamlit web interface.

## Features

* Image classification using Deep Learning
* Trained on the CIFAR-10 dataset
* Interactive Streamlit frontend
* Displays uploaded image
* Shows predicted class and confidence score
* Supports JPG, JPEG, and PNG image formats

## Dataset

The project uses the CIFAR-10 dataset, which contains:

* 60,000 color images
* 32 × 32 pixel resolution
* 10 object classes
* 50,000 training images
* 10,000 testing images

### Classes

* Airplane
* Automobile
* Bird
* Cat
* Deer
* Dog
* Frog
* Horse
* Ship
* Truck

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Pillow
* Streamlit
* Matplotlib

## Project Structure

Image_Classifier/

├── app.py

├── cifar10_image_classifier.h5

├── requirements.txt

├── README.md

└── notebooks/

## Installation

### Clone the Repository

git clone <repository-url>

cd Image_Classifier

### Install Dependencies

pip install -r requirements.txt

## Running the Application

streamlit run app.py

The application will open automatically in your browser.

## Model Architecture

* Convolutional Layer (32 Filters)
* Max Pooling Layer
* Convolutional Layer (64 Filters)
* Max Pooling Layer
* Convolutional Layer (64 Filters)
* Flatten Layer
* Dense Layer (64 Neurons)
* Output Layer (10 Classes with Softmax)

## Training Process

1. Load CIFAR-10 dataset
2. Normalize image pixel values
3. Build CNN architecture
4. Train model using training data
5. Evaluate model performance
6. Save trained model
7. Deploy using Streamlit

## Results

* Training Accuracy: Approximately 80–90%
* Testing Accuracy: Approximately 70–80%

Performance may vary depending on training parameters and number of epochs.

## Future Enhancements

* Data augmentation
* Transfer learning using pre-trained models
* Support for larger image datasets
* Improved model accuracy
* Cloud deployment

## Author

Developed as a Deep Learning and Computer Vision project demonstrating image classification using Convolutional Neural Networks and TensorFlow.
