# Watermelon Classification

## Overview

This project classifies watermelons into four categories: ripe, sweet, unripe, and watery. The classification is based on features such as shape, color, spot, and color of the spot.

## Features

- Image processing with OpenCV
- Feature extraction using various image properties
- Model training and evaluation using TensorFlow and scikit-learn
- Hyperparameter tuning with Keras Tuner
- Web application for classification using Flask
- Visualization of results with Matplotlib

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.x installed on your local machine
- The following Python libraries installed:
  - OpenCV
  - NumPy
  - shutil
  - random
  - TensorFlow
  - SciPy
  - scikit-learn
  - Keras Tuner
  - Flask
  - Matplotlib
  - PIL (Pillow)

## Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/omkarpawar201/Sweetscan.git
    cd Sweetscan
    ```

2. **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Set up the project directories:**
    - Ensure you have the appropriate directories for your dataset and models.

## Usage

1. **Prepare the dataset:**
    - Organize your images into directories based on their classification: ripe, sweet, unripe, and watery.
    - Update the dataset path in the script if necessary.

2. **Train the model:**
    - Run the training script to train the model on your dataset.
    ```sh
    python main.py
    ```


## Model Training

1. **Feature Extraction:**
    - Features such as shape, color, spot, and color of the spot are extracted from the images using OpenCV and NumPy.

2. **Model Training:**
    - The extracted features are used to train a classification model using TensorFlow.
    - Hyperparameters are tuned using Keras Tuner.
    - The trained model is saved for later use.

3. **Model Evaluation:**
    - The trained model is evaluated using various metrics from scikit-learn.


## Acknowledgements

- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [SciPy](https://www.scipy.org/)
- [scikit-learn](https://scikit-learn.org/)
- [Keras Tuner](https://keras-team.github.io/keras-tuner/)
- [Flask](https://flask.palletsprojects.com/)
- [Matplotlib](https://matplotlib.org/)
- [PIL (Pillow)](https://python-pillow.org/)
