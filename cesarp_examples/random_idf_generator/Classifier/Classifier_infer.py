import numpy as np
import pandas as pd
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Testing
# Step 1: Load the new data
new_inputs = np.load('/home/fazel/PycharmProjects/cesarp/cesarp_examples/random_idf_generator/results/example/final_tensor_test.npy')  # Replace with the actual path
new_targets = pd.read_csv('/home/fazel/PycharmProjects/cesarp/cesarp_examples/random_idf_generator/results/example/Categorized_Targets.csv').values  # Replace with the actual path
new_targets = new_targets[10000:20000, 0:4]  # targets[:10000, 8:16]

# Adjust the shape of new_inputs to be (1000, 365, 24, 9) (assuming 1000 new samples)
new_inputs = np.transpose(new_inputs, (3, 0, 1, 2))

assert new_inputs.shape == (10000, 365, 24, 9)
assert new_targets.shape == (10000, 4)

# Step 3: Preprocess the new data
scale_max = np.load('scale_max.npy')
new_inputs = new_inputs / scale_max  # Normalize the inputs

# Step 4: Evaluate the model on the new dataset
model = load_model('vgg_16.keras')
loss, accuracy = model.evaluate(new_inputs, new_targets, batch_size=100)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Step 5: Make predictions on the new dataset
predictions = model.predict(new_inputs)

# Convert predictions to class labels (if needed)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(new_targets, axis=1)

# Example: Compare the first 10 predictions to the true classes
for i in range(10):
    print(f"Sample {i}: Predicted = {predicted_classes[i]}, True = {true_classes[i]}")
