import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import dual_network_bounds
import matplotlib.pyplot as plt
import tensorflow as tf

### 2 inputs -> Dense layer 10 -> ReLU -> Dense Layer 10 -> ReLU -> Dense 2 outputs
W1 = np.random.uniform(-0.5,0.5,(2,100))
b1 = np.random.uniform(0.02,0.1,(100,1))

W2 = np.random.uniform(-0.5,0.5,(100,100))
b2 = np.random.uniform(0.02,0.1,(100,1))

W3 = np.random.uniform(-0.5,0.5,(100,100))
b3 = np.random.uniform(0.02,0.1,(100,1))

W4 = np.random.uniform(-0.5,0.5,(100,100))
b4 = np.random.uniform(0.02,0.1,(100,1))

W5 = np.random.uniform(-0.5,0.5,(100,2))
b5 = np.random.uniform(0.02,0.1,(2,1))

depth = 4
eps = 0.01

weights = [W1,b1,W2,b2,W3,b3,W4,b4,W5,b5] ###Keras weights (typical)
weights_t = []
for i in range(0, len(weights)):
    weights_t.append(tf.Variable(initial_value=weights[i].astype('float32'), trainable = True))

def dual_pred(weights, bounds, inputs, targets, eps):
    targets = targets.astype('float32')
    l = bounds[0]  
    u = bounds[1]
    S = np.size(targets, 0)
    targets_t = tf.convert_to_tensor(targets)
    inputs_t = tf.convert_to_tensor(inputs.astype('float32'))
    inputs_t = tf.reshape(inputs_t, (2,1))
    mul = tf.tensordot(targets_t , tf.transpose(tf.ones(S, 1)), axes = 0)
    mul = tf.reshape(mul, (S,S))
    nu_k = tf.eye(S)-mul ##Input to the dual neural network -c
    res =  - tf.transpose(nu_k) @ weights[2*depth+1] ##(2,1) dimension
    for i in range(2,depth+2):
        nu_hat = weights[2*(depth-i+2)] @ nu_k ##(10,2) shape 
        D_I = dual_network_bounds.form_D_I(l[depth-i+1],u[depth-i+1]) 
        D_I_t = tf.convert_to_tensor(D_I.astype('float32'))
        nu = D_I_t @ nu_hat ## (10, 2) shape!
        res = res - tf.transpose(nu)@weights[2*(depth-i+1)+1] ##(2,1) shape!
        for j in range(0, np.size(D_I,0)):
            if D_I[j][j] > 0 and D_I[j][j] < 1:
                nu_sl = tf.reshape(nu[j,:], (2,1))
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

bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM)

def grad(inputs, targets):
    Batch_Size = np.size(targets, 0)
    with tf.GradientTape() as tape:
        loss_value = 0
        for i in range(0, Batch_Size):
            bounds = dual_network_bounds.dual_network_bounds(weights, depth, eps, inputs[i][:])
            predict = dual_pred(weights_t, bounds, inputs[i][:], np.transpose(targets[i][:]), eps)
            target = tf.reshape(np.transpose(targets[i][:]).astype('float32'), (2,1))
            #target = tf.ones((2,1))-target
            predict = tf.exp(predict)/(tf.exp(predict[0])+tf.exp(predict[1]))
            loss_value = loss_value+bce(target, predict)
            #loss_value = loss_value+tf.exp(tf.transpose(predict)@tf.convert_to_tensor(target))
    return loss_value, tape.gradient(loss_value, weights_t)

###Set up experiment
N = 100
batches = 10 ###needs to divide N
x = np.random.uniform(0,1, (N, 2))
y = np.zeros((N,2))
epochs = 40
for i in range(0,N):
    success = False
    while success != True:
        if x[i][0]*x[i][0]+x[i][1]*x[i][1]>0.2 and x[i][0]*x[i][0]+x[i][1]*x[i][1]<0.8*0.8:
            y[i][0] = 1
            success = True
        elif x[i][0]*x[i][0]+x[i][1]*x[i][1]<0.15 or x[i][0]*x[i][0]+x[i][1]*x[i][1]>0.85*0.85:
            y[i][1] = 1
            success = True
        else: 
            x[i] = np.random.uniform(0,1,(1,2))          

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
best_weights = weights_t
best_loss = 1000000
for j in range(0,epochs):
    p = 0
    batches_size = int(N/batches)
    mean_loss = 0
    for i in range(0,batches):
        loss_value, grads = grad(x[10*i:10*i+batches_size][:], y[10*i:10*i+batches_size][:])
        optimizer.apply_gradients(zip(grads, weights_t))
        mean_loss = mean_loss+loss_value
        p = p+1
        if p > 0.1*batches:
            print(i/batches)
            p = 0
    #mean_loss = mean_loss/batches
    if mean_loss < best_loss:
        best_loss = mean_loss
        best_weights = weights_t
    else:
        weights_t = best_weights
        optimizer.learning_rate = 0.9*optimizer.learning_rate
    print("epoch "+str(j+1)+" :"+str(mean_loss))

def ypred(x, weights):
    y = np.zeros((2,1))
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
    y_pred_random = ypred(x[i][:], weights)
    y_pred_new_weights = ypred(x[i][:], weights_t_n)
    #print(y_pred_random)
    if y_pred_random[0] > y_pred_random[1] and y[i][0] == 1 or y_pred_random[0] < y_pred_random[1] and y[i][1] == 1:
        cor_r = cor_r+1
    if y_pred_new_weights[0] > y_pred_new_weights[1] and y[i][0] == 1 or y_pred_new_weights[0] < y_pred_new_weights[1] and y[i][1] == 1:
        cor_t = cor_t+1

print("random correct: "+str(cor_r/N))
print("new weights correct: "+str(cor_t/N))
# print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(), loss_value.numpy()))

#print("Step: {},         Loss: {}".format(optimizer.iterations.numpy(), grad(x, y)))
