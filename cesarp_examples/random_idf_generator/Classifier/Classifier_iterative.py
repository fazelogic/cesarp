import numpy as np
import pandas as pd
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Flatten, Dense, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import tensorflow as tf
import gc

# Step 1: Load the targets
targets = pd.read_csv('/home/fazel/PycharmProjects/cesarp/cesarp_examples/random_idf_generator/results/example/Categorized_Targets.csv').values  # Replace with the actual path
target_size = 5

# Step 2: Build a custom CNN model
def build_model(input_shape, target_size):
    inputs = Input(shape=input_shape)

    x = ZeroPadding2D(padding=1)(inputs)
    x = Conv2D(16, (12, 4), strides=(1, 1), padding='same', activation='elu')(x)
    x = AveragePooling2D((2, 2))(x)

    x = Conv2D(8, (6, 6), strides=(2, 2), padding='same', activation='elu')(x)
    x = AveragePooling2D((2, 2))(x)

    x = Conv2D(8, (3, 3), strides=(2, 2), padding='same', activation='elu')(x)
    x = AveragePooling2D((1, 1))(x)

    x = Flatten()(x)
    x = Dense(256, activation='elu')(x)
    output = Dense(target_size, activation='softmax')(x)

    model = Model(inputs, output)
    return model

# Step 3: Compile the model
input_shape = (365, 24, 9)  # Define the input shape
model = build_model(input_shape, target_size)
model.compile(optimizer=Adam(0.0005, 0.9), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Step 4: Train the model
num_batches = 9
batch_size = 10000
epochs_per_batch = 5
num_repeats = 1000

for repeat in range(num_repeats):
    print(f"Repeat {repeat + 1}/{num_repeats}")
    for batch_idx in range(1, num_batches + 1):
        # Load the inputs for the current batch
        inputs = np.load(f'/home/fazel/PycharmProjects/cesarp/cesarp_examples/random_idf_generator/results/example/final_tensor_{(batch_idx - 1) * batch_size + 1}-{batch_idx * batch_size}.npy')
        inputs = np.transpose(inputs, (3, 0, 1, 2))  # Move the last axis to the front

        # Normalize the inputs
        scale_max = np.max(inputs)
        inputs = inputs / scale_max

        # Select the corresponding targets for the current batch
        targets_batch = targets[(batch_idx - 1) * batch_size:batch_idx * batch_size, 20:25]

        # Ensure the shape of inputs and targets is as expected
        assert inputs.shape == (batch_size, 365, 24, 9)
        assert targets_batch.shape == (batch_size, target_size)

        # Train the model on the current batch
        history = model.fit(inputs, targets_batch, epochs=epochs_per_batch, batch_size=5000, verbose=1)

        # Clear memory after training on the current batch
        del inputs, targets_batch
        gc.collect()
        tf.keras.backend.clear_session()

# Step 5: Save the model
model.save('cnn.keras')

# Step 6: Plot the train-test errors (using the last history object from the final batch)
# plt.plot(history.history['loss'], label='train loss')
# plt.title('Train Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
#
# plt.plot(history.history['accuracy'], label='train accuracy')
# plt.title('Train Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()