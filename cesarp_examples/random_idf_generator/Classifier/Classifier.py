import numpy as np
import pandas as pd
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Flatten, Dense, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Step 1: Load the data
inputs = np.load('/home/fazel/PycharmProjects/cesarp/cesarp_examples/random_idf_generator/results/example/final_tensor_1-10000.npy')  # Replace with the actual path
targets = pd.read_csv('/home/fazel/PycharmProjects/cesarp/cesarp_examples/random_idf_generator/results/example/Categorized_Targets.csv').values  # Replace with the actual path
targets = targets[:10000, 20:25]  # targets[:10000, 8:16]
target_size = 5

# Adjust the shape of inputs to be (1000, 365, 24, 9)
inputs = np.transpose(inputs, (3, 0, 1, 2))  # Move the last axis to the front

# Ensure the shape of inputs and targets is as expected
assert inputs.shape == (10000, 365, 24, 9)
assert targets.shape == (10000, target_size)

# Step 2: Preprocess the data
# Normalize the inputs if necessary
scale_max = np.max(inputs)
np.save('scale_max', scale_max)

inputs = inputs / scale_max

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.2, random_state=42)

# Step 4: Build a custom CNN model
input_shape = X_train.shape[1:]  # (365, 24, 9)

# Define the model
inputs = Input(shape=input_shape)

x = ZeroPadding2D(padding=1)(inputs)
x = Conv2D(16, (12, 4), strides=(1, 1), padding='same', activation='elu')(x)
x = AveragePooling2D((2, 2))(x)

x = Conv2D(8, (6, 6), strides=(2, 2), padding='same', activation='elu')(x)
x = AveragePooling2D((2, 2))(x)

x = Conv2D(8, (3, 3), strides=(2, 2), padding='same', activation='elu')(x)
x = AveragePooling2D((1, 1))(x)

# x = Concatenate()([x, x1, x2, x])

x = Flatten()(x)
x = Dense(256, activation='elu')(x)
output = Dense(target_size, activation='softmax')(x)

model = Model(inputs, output)

# Step 5: Compile the model
model.compile(optimizer=Adam(0.0005, 0.9), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Step 6: Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1000, batch_size=5000)
model.save('vgg_16.keras')

# Step 7: Plot the train-test errors
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='test loss')
plt.title('Train vs Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='test accuracy')
plt.title('Train vs Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

