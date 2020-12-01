import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from timeit import default_timer as timer

weights = np.load('weights_test_2.npy', allow_pickle=True)

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

x_train = X_train.reshape(60000, 784).astype("float32") / 255
x_test =X_test.reshape(10000, 784).astype("float32") / 255

def relu(X):
    return np.maximum(0,X)

def predict(input):
    l1 = relu(np.matmul(input, np.transpose(weights[0]))+np.transpose(weights[1]))
    l2 = relu(np.matmul(l1,np.transpose(weights[2]))+np.transpose(weights[3]))
    l3 = np.matmul(l2, np.transpose(weights[4]))+np.transpose(weights[5])
    return l3

gamma = 10
L = 49
a = gamma/L

x = x_train[1,:] + np.random.uniform(0,a, (1000, 784))
begin = timer()
y = predict(x)
print(y[0])
print(y[1])
end = timer()
cor = 0
print("time "+str(end-begin))
for i in range(0,1000):
    if np.argmax(y[i]) == 0:
        cor = cor+1
print(cor/1000)
plt.figure
plt.imshow(X_train[1], cmap='gray')
plt.show()

