import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import dual_network_bounds
import dual_network_bounds_tensorflow
import matplotlib.pyplot as plt
import tensorflow as tf
import  matplotlib.patches as patches

### conv_net
model = keras.Sequential()
model.add(keras.Input(shape=(784,1)))
##model.add(keras.layers.Conv1D(16, 1, strides=2, activation="relu"))
##model.add(keras.layers.Conv1D(32, 1, strides=2, activation="relu"))
model.add(keras.layers.Dense(100, activation= "relu")) ##First try
model.add(keras.layers.Dense(100, activation= "relu"))
model.add(keras.layers.Dense(10))
model.compile(optimizer='adam', loss='categorical_crossentropy')

W1 = np.random.normal(0,1/np.sqrt(784),(784,100))
b1 = np.random.normal(0,1,(100,1))

W2 = np.random.normal(0,1/np.sqrt(100),(100,100))
b2 = np.random.normal(0,1,(100,1))

W3 = np.random.normal(0,1/np.sqrt(100),(100,10))
b3 = np.random.normal(0,1,(10,1))

depth = 2
eps = 0.05

weights_r = [W1,b1,W2,b2,W3,b3] ###Keras weights
weights_t = []
for i in range(0, len(weights_r)):
    weights_t.append(tf.Variable(initial_value=weights_r[i].astype('float32'), trainable = True))

weights_np_t = np.copy(weights_r) ###The changing numpy training weights!
def dual_pred(weights, bounds, inputs, targets, eps):
    targets = targets.astype('float32')
    l = bounds[0]
    u = bounds[1]
    S = np.size(targets, 0)
    targets_t = tf.convert_to_tensor(targets)
    inputs_t = tf.convert_to_tensor(inputs.astype('float32'))
    inputs_t = tf.reshape(inputs_t, (784,1))
    mul = tf.tensordot(targets_t , tf.transpose(tf.ones(S, 1)), axes = 0)
    mul = tf.reshape(mul, (S,S))
    nu_k = tf.eye(S)-mul ##Input to the dual neural network -c
    res =  - tf.transpose(nu_k) @ weights[2*depth+1] ##(2,1) dimension
    for i in range(2,depth+2):
        nu_hat = weights[2*(depth-i+2)] @ nu_k ##(10,2) shape 
        D_I_t = dual_network_bounds_tensorflow.form_D_I_t(l[depth-i+1],u[depth-i+1]) 
        #D_I_t = tf.convert_to_tensor(D_I.astype('float32'))
        nu = D_I_t @ nu_hat ## (10, 2) shape!
        res = res - tf.transpose(nu)@weights[2*(depth-i+1)+1] ##(2,1) shape!
        for j in range(0, tf.shape(D_I_t)[0]):
            if D_I_t[j,j] > 0 and D_I_t[j,j] < 1:
                nu_sl = tf.reshape(nu[j,:], (10,1))
                nu_sl = (tf.abs(nu_sl)+nu_sl)/2 ##Take positive part
                res = res + tf.math.multiply(l[depth-i+1][0][j],nu_sl)
        nu_k = nu 
        ###Here res still (2, 1) shape
    nu_hat_1 = weights[0] @ nu_k
    res = res - tf.transpose(nu_hat_1) @ inputs_t
    S_n = tf.shape(nu_hat_1)
    nu_hat_norm = tf.Variable(initial_value = tf.zeros((S_n[1],1)), trainable = False)
    for i in range(0, S_n[1]):
        for j in range(0, S_n[0]):
            nu_hat_norm[i].assign(tf.abs(nu_hat_1[j,i])+nu_hat_norm[i])
    res = res - eps*nu_hat_norm
    return -res

ce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM, from_logits=True)

def grad(inputs, targets):
    Batch_Size = np.size(targets, 0)
    with tf.GradientTape() as tape:
        loss_value = 0
        for i in range(0, Batch_Size):
            #bounds = dual_network_bounds_tensorflow.dual_network_bounds_t(weights_t, depth, eps, tf.convert_to_tensor(inputs[i][:].astype('float32')))
            bounds = dual_network_bounds.dual_network_bounds(weights_np_t, depth, eps, inputs[i][:])
            predict = dual_pred(weights_t, bounds, inputs[i][:], np.transpose(targets[i][:]), eps)
            target = tf.reshape(np.transpose(targets[i][:]).astype('float32'), (10,1))
            loss_value = loss_value+ce(target, predict)/Batch_Size
            #loss_value = loss_value+tf.exp(10*tf.transpose(predict)@tf.convert_to_tensor(targett))
    return loss_value, tape.gradient(loss_value, weights_t)

###Set up experiment
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
N = np.size(x_train, 0)
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))
x = x_train
y = keras.utils.to_categorical(y_train)
#batches = int(N/50)
batches = N
print("number of batches: "+str(batches))
epochs = 10
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
#optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
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
        loss_value, grads = grad(x[batches_size*i:batches_size*i+batches_size][:], y[batches_size*i:batches_size*i+batches_size][:])
        optimizer.apply_gradients(zip(grads, weights_t))
        for m in range(0, len(weights_t)):
            weights_np_t[m] = tf.identity(weights_t[m]).numpy()
        mean_loss = mean_loss+loss_value/batches
        p = p+1
        print("batch number done: "+str(i))
        if p > 0.01*batches:
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

def ypred(x, weights):
    y = np.zeros((10,1))
    for i in range(0, depth):
        y = np.matmul(x, weights[2*i])+np.transpose(weights[2*i+1])
        y = y[0]
        for j in range(0, np.size(y,0)):
            if y[j] < 0:
                y[j] = 0
        x = y
    y = np.matmul(y, weights[2*depth])+np.transpose(weights[2*depth+1])
    return np.transpose(y)

weights_t_n = []
for i in best_weights:
    weights_t_n.append(i.numpy())

cor_r = 0
cor_t = 0
for i in range(0, N):
    y_pred_random = ypred(x[i][:], weights_r)
    y_pred_new_weights = ypred(x[i][:], weights_t_n)
    #print(y_pred_random)
    if y_pred_random[0] > y_pred_random[1] and y[i][0] == 1 or y_pred_random[0] < y_pred_random[1] and y[i][1] == 1:
        cor_r = cor_r+1
    if y_pred_new_weights[0] > y_pred_new_weights[1] and y[i][0] == 1 or y_pred_new_weights[0] < y_pred_new_weights[1] and y[i][1] == 1:
        cor_t = cor_t+1

print("random correct: "+str(cor_r/N))
print("new weights correct: "+str(cor_t/N))
