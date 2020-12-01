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

timer_g = 0

ce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM, from_logits=True)
depth = 2
M = 16
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
epochs = 10
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
#optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

def predictions_t(inputs, weights_t):
    results = tf.nn.relu(weights_t[0]@tf.transpose(inputs)+weights_t[1])
    for i in range(1,depth):
        results = tf.nn.relu(weights_t[2*i]@results+weights_t[2*i+1])
    results = weights_t[2*depth]@results+weights_t[2*depth+1]
    return results

###Mullers methods
def Mullers_sample_of_d_sphere(r, Batch_Size):
    x_noise = np.zeros((Batch_Size, 28,28))
    u = np.random.normal(0,1,(Batch_Size, 28,28))  # an array of d normally distributed random variables
    for i in range(0,4):
        for j in range(0,4):
            norm=np.reshape(np.linalg.norm(u[:, i*7:i*7+7,j*7:j*7+7] , axis = (1,2)),(Batch_Size, 1))
            r_n = np.random.uniform(0,1, (Batch_Size,1))**(1.0/49)
            x_inter = np.multiply(r_n, np.true_divide(np.reshape(u[:, i*7:i*7+7,j*7:j*7+7],(Batch_Size, 49)), norm))
            x_noise[:, i*7:i*7+7,j*7:j*7+7] = np.reshape(x_inter, (Batch_Size,7,7))
    return r*np.reshape(x_noise, (Batch_Size, 784))


###Dropped coordinates
def dropped_coord(r,N):
    u = np.random.normal(0,1,N+2)  # an array of (d+2) normally distributed random variables
    norm=np.sum(u**2) **(0.5)
    u = u/norm
    x = r*u[0:N] #take the first d coordinates
    return np.reshape(x, (7,7))

def add_noise(r, Batch_Size):
    x_noise = Mullers_sample_of_d_sphere(r, Batch_Size)
    print(x_noise)
    return x_noise.astype('float32')

def loss_IBP(inputs, targets):
    return 0

@tf.function
def loss_adv(x, y, r, Batch_Size):
    x = x + add_noise(r, Batch_Size).astype('float32')
    ###Now FCGS step
    y_pred = tf.transpose(predictions_t(x, weights_t))
    return ce(y, y_pred)

def grad_IBP(inputs, targets):
    Batch_Size = np.size(targets, 0)
    #input_t = tf.convert_to_tensor(inputs.astype('float32'))
    #targets_t = tf.convert_to_tensor(targets.astype('float32'))
    with tf.GradientTape() as tape:
        loss_value = 0
        #predictions = predictions_t(input_t, weights_t)
        for i in range(0, Batch_Size):
            loss_value = loss_value + loss_IBP(inputs[i], targets[i]) 
    return loss_value, tape.gradient(loss_value, weights_t)

def grad_adv(inputs, targets, r):
    Batch_Size = np.size(targets, 0)
    #input_t = tf.convert_to_tensor(inputs.astype('float32'))
    #targets_t = tf.convert_to_tensor(targets.astype('float32'))
    with tf.GradientTape() as tape:
        #predictions = predictions_t(input_t, weights_t)
        loss_value = loss_adv(inputs, targets, r, Batch_Size) 
    return loss_value/Batch_Size, tape.gradient(loss_value, weights_t)

best_weights = weights_t
best_loss = 100000000000
worsened = 0
print("started training")
for j in range(0,epochs):
    print("start epoch "+str(j+1))
    p = 0
    batches_size = int(N/batches)
    mean_loss = 0
    for i in range(0,batches):
        #begin = timer()
        r = 1.5
        loss_value, grads = grad_adv(x_train[batches_size*i:batches_size*i+batches_size][:], y[batches_size*i:batches_size*i+batches_size][:], r)
        optimizer.apply_gradients(zip(grads, weights_t))
        #end = timer()
        #print("time for batch: "+str(end-begin))
        mean_loss = mean_loss+loss_value/batches
        p = p+1
        if p > 0.1*batches:
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

y_pred = predictions_t(x_test, weights_t)
score = 0
for i in range(10000):
    if tf.math.argmax(y_pred[:,i]) == Y_test[i]:
        score = score+1
print("test accuracy: "+str(score/10000))
for i in range(len(weights)):
    weights[i] = weights_t[i].numpy()
np.save('adv_IBP_weights_test_1.npy', weights)
