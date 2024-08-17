
# MNIST CNN Model: Training, Quantization, and Evaluation

This project demonstrates how to train a Convolutional Neural Network (CNN) on the MNIST dataset, quantize the trained model to INT8 using TensorFlow Lite, and evaluate the performance of the quantized model.

## Project Overview
The project consists of two Python scripts:

- model.py: Handles a CNN model's creation, training, and quantization.
- evaluate.py: Evaluates the performance of the quantized INT8 model.

### 1. model.py
This script performs the following steps:

- Model Definition: Defines a CNN architecture tailored for MNIST digit classification.
- Data Preparation: Loads and preprocesses the MNIST dataset, normalizing the pixel values to the range [0, 1].
- Model Training: Trains the CNN model on the MNIST training dataset for 5 epochs.
- Model Evaluation: Evaluates the model on the test dataset and reports the accuracy.
- Model Saving: Saves the trained model in full precision (FP32) format as cnn_fp32_model.
- Model Quantization: Converts the FP32 model to an INT8 model using TensorFlow Lite, incorporating a representative dataset to optimize the quantization process while maintaining input and output in FP32.
- Quantized Model Saving: Saves the quantized model as cnn_int8_model.tflite.

### 2. evaluate.py
This script is responsible for:

- Model Loading: Loads the previously saved INT8 TensorFlow Lite model (cnn_int8_model.tflite).
- Dataset Preparation: Loads and preprocesses the MNIST test dataset.
- Model Evaluation: Runs inference on the test dataset using the quantized model and calculates the accuracy.
- Results Reporting: Prints the accuracy of the INT8 model, allowing comparison with the original FP32 model.

## Installation and Setup

### 1. Environment Setup
Before running the scripts, ensure your environment is properly configured.

- Step 1: Clone the Repository
```
git clone https://github.com/OpenGenus/int8-training.git
cd int8-training
```

- Step 2: Set Up the Environment (Install dependencies)

```
pip install NumPy
pip install TensorFlow
```

### 2. Running the Scripts

**To train and quantize the model, run:**

```python3 model.py```

This script will:

- Train a CNN on the MNIST dataset.
- Save the trained FP32 model as cnn_fp32_model.
- Quantize the model to INT8 and save it as cnn_int8_model.tflite.

**To evaluate the performance of the quantized model, run:**

```python3 evaluate.py```

This script will:

- Load the cnn_int8_model.tflite file.
- Run inference on the MNIST test dataset.
- Print the accuracy of the quantized model.

## Project Structure
```
├── model.py         # Script for training and quantizing the CNN model
├── evaluate.py      # Script for evaluating the quantized model
└──  README.md        # Project documentation (this file)
```

## Acknowledgements
This project leverages the TensorFlow and Keras libraries for deep learning and model quantization. The MNIST dataset is a well-known dataset of handwritten digits, widely used for training and evaluating image processing systems.

## Authors

**Vidhi Srivastava**
- [GitHub](https://github.com/Vidhi0229)
- [LinkedIn](https://www.linkedin.com/in/vidhisrivastava01/)

## Show Your Support ⭐️⭐️
If you find this project helpful or interesting, please consider giving it a star!
