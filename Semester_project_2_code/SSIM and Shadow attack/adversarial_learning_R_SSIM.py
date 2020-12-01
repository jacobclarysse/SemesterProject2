import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from timeit import default_timer as timer

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

x_train = X_train.reshape(60000, 784).astype("float32") / 255
x_test =X_test.reshape(10000, 784).astype("float32") / 255

ce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM, from_logits=True)
depth = 2
M = 4
gamma = 1.8
L = 49
a = gamma/L
y = keras.utils.to_categorical(Y_train)
y_test = keras.utils.to_categorical(Y_test)
N = 60000
W1 = np.random.normal(0,1/np.sqrt(784),(100,784))
b1 = 0.1*np.ones((100,1))
W2 = np.random.normal(0,0.1,(100,100))
b2 = 0.1*np.ones((100,1))
W3 = np.random.normal(0,0.1,(10,100))
b3 = 0.1*np.ones((10,1))
weights = [W1,b1,W2,b2,W3,b3]
weights_t = []
for i in range(0, len(weights)):
    weights_t.append(tf.Variable(initial_value=weights[i].astype('float32'), trainable = True))
batches = 1000
print("number of batches: "+str(batches))
epochs = 50
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
#optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

def predictions_t(inputs, weights_t):
    results = tf.nn.relu(weights_t[0]@tf.transpose(inputs)+weights_t[1])
    for i in range(1,depth):
        results = tf.nn.relu(weights_t[2*i]@results+weights_t[2*i+1])
    results = weights_t[2*depth]@results+weights_t[2*depth+1]
    return results

def random_noise(inputs):
    inputs = inputs+tf.random.uniform((60,784),0,a)
    return inputs

def special_norm_matrice(matrix):
    norms = []
    S = tf.shape(matrix)[0]
    for j in range(0, S):
        v = tf.reshape(matrix[j,:],(28,28))
        m_e = 0
        for i in range(M):
            m = tf.norm(v[7*i:7*i+7,7*i:7*i+7])
            if m > m_e:
                m_e = m
        norms.append(m_e)
    return norms

@tf.function
def grad(inputs, targets):
    #Batch_Size = np.size(targets, 0)
    #input_t = tf.convert_to_tensor(inputs.astype('float32'))
    #targets_t = tf.convert_to_tensor(targets.astype('float32'))
    with tf.GradientTape() as tape:
        loss_value = 0
        predictions = predictions_t(random_noise(inputs), weights_t)
        #loss_value = (ce(tf.transpose(targets_t), predictions)+MMR_v)/Batch_Size
        loss_value = ce(targets, tf.transpose(predictions))
    return loss_value, tape.gradient(loss_value, weights_t)

best_weights = weights_t
best_loss = 100000000000
worsened = 0
print("started training")
t = 0
for j in range(0,epochs):
    print("start epoch "+str(j+1))
    p = 0
    batches_size = int(N/batches)
    mean_loss = 0
    for i in range(0,batches):
        start = timer()
        loss_value, grads = grad(x_train[batches_size*i:batches_size*i+batches_size][:], y[batches_size*i:batches_size*i+batches_size][:])
        optimizer.apply_gradients(zip(grads, weights_t))
        end = timer()
        t += end-start
        mean_loss = mean_loss+loss_value/batches
        p = p+1
        if p > 0.5*batches:
            print(i/batches)
            p = 0
    #mean_loss = mean_loss/batches
    if mean_loss < best_loss:
        best_loss = mean_loss
        best_weights = weights_t.copy()
        worsened = 0
    else:
        a = 1
        worsened = worsened + 1
        weights_t = best_weights
        if worsened == 11:
            break
        #optimizer.learning_rate = 0.9*optimizer.learning_rate
    print("epoch "+str(j+1)+" :"+str(mean_loss))

print("total training time: "+str(t))
y_pred = predictions_t(x_test, weights_t)
score = 0
for i in range(10000):
    if tf.math.argmax(y_pred[:,i]) == Y_test[i]:
        score = score+1
print("test accuracy: "+str(score/10000))
for i in range(len(weights)):
    weights[i] = weights_t[i].numpy()
np.save('weights_test_2.npy', weights)