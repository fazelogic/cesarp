import os
import cesarp.common
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def __abs_path(path):
    return cesarp.common.abs_path(path, os.path.abspath(__file__))

read_path = __abs_path(f"./results/example/final_tensor.npy")
# Load the input data (tensor) and target labels
input_tensor = np.load(read_path)  # Shape: [m, n, p, q]
read_path = __abs_path(f"./results/example/Categorized_Targets.csv")
target = pd.read_csv(read_path).values  # Shape: [q, t]
target = target[:1000, :]


# Reshape the input tensor
m, n, p, q = input_tensor.shape
X = input_tensor.reshape(m * n * p, q).T  # Shape: [q, m*n*p]

# Flatten the target if it has multiple categories
# Assuming the target is multi-label classification (one-hot encoded)
y = target.argmax(axis=1)  # Convert from one-hot to single label (if needed)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Gradient Boosting Classifier
model = HistGradientBoostingClassifier()

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

outputs = pd.DataFrame(y_pred)
outputs.to_csv('xgboost_outputs.csv')

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
