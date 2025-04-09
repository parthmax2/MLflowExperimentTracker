import pandas as pd
import numpy as np
import random
import mlflow.sklearn
from sklearn.datasets import load_iris

# Load the Iris dataset again (yeah we need this to make sense of labels/features)
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# Load the model that we saved and registered earlier
model = mlflow.sklearn.load_model("models:/IrisRandomForestModel/1")  # you can change version if needed

# Grab a random index from the dataset for testing
rand_index = random.randint(0, len(X) - 1)

# Get input and actual label from dataset
sample_input = X.iloc[rand_index].values.reshape(1, -1)
sample_true = y.iloc[rand_index]

# Predict using the loaded model
sample_pred = model.predict(sample_input)[0]

# Print stuff to feel like a proper data scientist 😎
print("\nRandom Sample Test:")
print("Input features:", sample_input)
print("True label:", iris.target_names[sample_true])
print("Predicted label:", iris.target_names[sample_pred])
