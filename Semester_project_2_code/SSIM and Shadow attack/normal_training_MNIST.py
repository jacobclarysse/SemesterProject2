import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

x_train = X_train.reshape(60000, 784).astype("float32") / 255
x_test =X_test.reshape(10000, 784).astype("float32") / 255

input = keras.Input(shape=(784,))
dense_1 = layers.Dense(150, activation="relu")
l1 = dense_1(input)
l2 = layers.Dense(150, activation="relu")(l1)
output = layers.Dense(10)(l2)
model = keras.Model(inputs=input, outputs=output, name="mnist_model_test_1")
model.summary()

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    metrics=["accuracy"],
)

history = model.fit(x_train, Y_train, batch_size=64, epochs=50)

test_scores = model.evaluate(x_test, Y_test, verbose=2)
model.save('MNIST_standard.h5')
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])