# Diabetes Prediction System

This is a simple Python script for building a diabetes prediction system using machine learning. The code uses the scikit-learn library to train a Random Forest Classifier on a diabetes dataset, and it then evaluates the model's accuracy.

## Getting Started

To get started with this project, you need to have Python installed on your machine, along with the required libraries (pandas, numpy, scikit-learn). You can install these libraries using `pip` if you haven't already:

```bash
pip install pandas numpy scikit-learn
```

## Code Overview

This project consists of a Python script that performs the following steps:

1. **Load Data**: It loads the diabetes dataset from a CSV file using the `pandas` library.

2. **Data Preprocessing**:
   - It separates the features (X) and the target variable (y).
   - The data is split into training and testing sets (80% training, 20% testing) using `train_test_split` from scikit-learn.
   - Feature scaling is applied using `StandardScaler` to standardize the feature values.

3. **Model Training**:
   - A Random Forest Classifier is chosen as the machine learning model and trained using the training data.

4. **Model Evaluation**:
   - The trained model is used to make predictions on the test data.
   - The accuracy of the model is calculated using `accuracy_score` from scikit-learn, and the result is printed.

## Usage

You can run the provided Python script to train the diabetes prediction model and evaluate its accuracy. To do so, follow these steps:

1. Download the diabetes dataset (in CSV format) and save it to your local machine. You should specify the path to the dataset in the `data` variable.

2. Open a command prompt or terminal and navigate to the directory where the script is located.

3. Run the script using Python:

```bash
python diabetes_prediction.py
```

The script will load the dataset, preprocess the data, train the model, and output the accuracy of the model on the test data.