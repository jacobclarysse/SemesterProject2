import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from timeit import default_timer as timer
import utils

weights = np.load('weights_test_4.npy', allow_pickle=True)
weights_t = []
for i in weights:
    weights_t.append(tf.convert_to_tensor(i))


(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

y_train = keras.utils.to_categorical(Y_train)
y_test = keras.utils.to_categorical(Y_test)
x_train = X_train.reshape(60000, 784).astype("float32") / 255
x_test =X_test.reshape(10000, 784).astype("float32") / 255
cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM, from_logits=True)

gamma = 1.5
L = 49
a = gamma/L
M = 16
lambda_tv = 0.1
lambda_c = 0.001
steps = 1000
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
norms = utils.special_norm_matrice(weights[0], 100)
maxi = 0
max_iter=0
r = 0.1
max_cer = 0
for i in range(10000):
    m = utils.best_certificate(x_test[i],norms,y_test[i], weights, M)
    if m>r:
        maxi += 1
        if m > max_cer:
            max_cer=m
            max_iter = i
    if i%1000 == 0:
        print(i)

print(maxi)
print(max_cer)
print(max_iter)

image_t = tf.convert_to_tensor(x_train[4263].astype("float32"))
initial_val = np.random.normal(0,0.1, (784,1))
delta = tf.Variable(initial_value=np.reshape(initial_val,(1,784)).astype("float32"), trainable = True)
delta_l = [delta]

#get best certificate
##4263, 0.2792623

###Shadow attack
def loss(img, delta, weights):
    y_pred = utils.predict_t(img+delta, weights)
    y_true = tf.convert_to_tensor(np.array([0,0,1,0,0,0,0,0,0,0]).transpose().astype("float32"))
    y_pred = tf.reshape(y_pred,(10,1))
    print(cce(y_true, y_pred))
    loss_v = cce(y_true, y_pred)+lambda_tv*utils.TV(delta)+lambda_c*utils.c(delta)
    return loss_v

def grad(img, delta_l, weights_t):
    with tf.GradientTape() as tape:
        loss_value = loss(img, delta_l[0], weights_t)
    return loss_value, tape.gradient(loss_value, delta_l)

#for i in range(steps):
#   loss_value, grads = grad(image_t, delta_l, weights_t)
#   optimizer.apply_gradients(zip(grads, delta_l))

#image_new = np.clip((x_train[4263]+delta_l[0].numpy()), 0, 1)
#print(utils.predict(image_new,weights))
#image_new = image_new[0]
#image_new = np.reshape(image_new, (28,28)) 
#plt.figure(1)
#plt.imshow(image_new, cmap='gray')

#plt.figure(2)
#plt.imshow(X_train[4263], cmap='gray')
#plt.show()

