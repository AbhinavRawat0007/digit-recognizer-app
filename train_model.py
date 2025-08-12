# train_model.py

import tensorflow as tf

# 1. Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Preprocess the data
# Reshape data to fit the model (add a channel dimension)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Normalize pixel values from [0, 255] to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

print("Data loaded and preprocessed.")

# 3. Define the CNN model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 4. Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Model compiled. Starting training...")

# 5. Train the model
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

# 6. Evaluate the model (optional)
loss, acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nModel accuracy on test data: {acc*100:.2f}%")

# 7. Save the trained model
model.save('mnist_model.h5')
print("\nModel saved successfully as 'mnist_model.h5'")