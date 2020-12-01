import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import dual_network_bounds
import dual_network_bounds_tensorflow
import matplotlib.pyplot as plt
import tensorflow as tf
import  matplotlib.patches as patches

### Feedforward neural network
W1 = np.random.normal(0,1/np.sqrt(2),(2,100))
b1 = np.random.normal(0,1,(100,1))

W2 = np.random.normal(0,1/np.sqrt(100),(100,100))
b2 = np.random.normal(0,1,(100,1))

W3 = np.random.normal(0,1/np.sqrt(100),(100,100))
b3 = np.random.normal(0,1,(100,1))

W4 = np.random.normal(0,1/np.sqrt(100),(100,100))
b4 = np.random.normal(0,1,(100,1))

W5 = np.random.normal(0,1/np.sqrt(100),(100,2))
b5 = np.random.normal(0,1,(2,1))

depth = 4
eps = 0.08

weights_r = [W1,b1,W2,b2,W3,b3,W4,b4,W5,b5] ###Keras weights (typical)
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
    inputs_t = tf.reshape(inputs_t, (2,1))
    mul = tf.tensordot(targets_t , tf.transpose(tf.ones(S, 1)), axes = 0)
    mul = tf.reshape(mul, (S,S))
    nu_k = tf.eye(S)-mul ##Input to the dual neural network -c
    res =  - tf.transpose(nu_k) @ weights[2*depth+1] ##(2,1) dimension
    for i in range(2,depth+2):
        nu_hat = weights[2*(depth-i+2)] @ nu_k ##(10,2) shape 
        D_I_t, indexes = dual_network_bounds_tensorflow.form_D_I_t_2(l[depth-i+1],u[depth-i+1]) 
        #D_I_t = tf.convert_to_tensor(D_I.astype('float32'))
        nu = D_I_t @ nu_hat ## (10, 2) shape!
        res = res - tf.transpose(nu)@weights[2*(depth-i+1)+1] ##(2,1) shape!
        for j in indexes:
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
        nu_hat_norm[i].assign(tf.reshape(tf.norm(nu_hat_1[:,i], ord=1), (1,)))
    res = res - eps*nu_hat_norm
    return -res

bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM, from_logits=True)

def grad(inputs, targets):
    Batch_Size = np.size(targets, 0)
    #cor = 0
    with tf.GradientTape() as tape:
        loss_value = 0
        for i in range(0, Batch_Size):
            #print("before bounds")
            #bounds = dual_network_bounds_tensorflow.dual_network_bounds_t(weights_t, depth, eps, tf.convert_to_tensor(inputs[i][:].astype('float32')))
            #print("after bounds")
            #print(bounds[0][0])
            bounds = dual_network_bounds.dual_network_bounds(weights_np_t, depth, eps, inputs[i][:])
            #print(bounds_1[0][0]-bounds[0][0])
            predict = dual_pred(weights_t, bounds, inputs[i][:], np.transpose(targets[i][:]), eps)
            target = tf.reshape(np.transpose(targets[i][:]).astype('float32'), (2,1))
            #targett = tf.ones((2,1))-target
            #if tf.transpose(predict)@tf.convert_to_tensor(targett)<0:
            #    cor = cor + 1
            loss_value = loss_value+bce(target, predict)/Batch_Size
            #loss_value = loss_value+tf.exp(10*tf.transpose(predict)@tf.convert_to_tensor(targett))
    #print(cor)
    return loss_value, tape.gradient(loss_value, weights_t)

###Set up experiment
N = 5
batches = 1 ###needs to divide N
x = np.random.uniform(0,1, (N, 2))
y = np.zeros((N,2))
epochs = 50
print("start finding points")
for i in range(0,N):
    success = False
    print(i)
    while success != True:
        if i == 0:
            y_t = np.random.uniform(0,1,1)
            if y_t > 0.5:
                y[i][0] = 1
                success = True
            else:
                y[i][1] = 1
                success = True
        else:
            success = True
            for j in range(0, i):
                if np.abs(x[j][0]-x[i][0]) < 0.24 and np.abs(x[j][1]-x[i][1]) < 0.24:
                    success = False
                    x[i][:] = np.random.uniform(0,1, (1, 2))
                    break
            if success == True:
                y_t = np.random.uniform(0,1,1)
                if y_t > 0.5:
                    y[i][0] = 1
                else:
                    y[i][1] = 1
                
print("Found points!")
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
#optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
best_weights = weights_t
best_loss = 10000000
worsened = 0
for j in range(0,epochs):
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
        if p > 0.1*batches:
            #print(i/batches)
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
    y_pred_random = ypred(x[i][:], weights_r)
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
#with open('robust_net_toy', 'wb') as fp:
#    pickle.dump(weights_np_t, fp)
model = keras.Sequential()
model.add(keras.Input(shape=(2,)))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(2, activation="relu"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=200)


fig1, ax1 = plt.subplots(1)
for i in range(0,N):
    if y[i][0] > y[i][1]:
        plt.plot(x[i][0],x[i][1], "r*")
        rect = patches.Rectangle((x[i][0]-0.08, x[i][1]-0.08), 0.16, 0.16,edgecolor='r', facecolor="none")
        ax1.add_patch(rect)
    else:
        plt.plot(x[i][0],x[i][1], "b*")
        rect = patches.Rectangle((x[i][0]-0.08, x[i][1]-0.08), 0.16, 0.16,edgecolor='b', facecolor="none")
        ax1.add_patch(rect)
x_ax = np.linspace(0,1, 20)
y_ax = np.linspace(0,1, 20)
for i in range(0, 20):
    for j in range(0, 20):
        x_in = np.array([x_ax[i], y_ax[j]])
        pred = ypred(x_in, weights_t_n)
        if pred[0] > pred[1]:
            plt.plot(x_in[0], x_in[1], 'y+')
        else:
            plt.plot(x_in[0], x_in[1], 'b+')

fig2, ax2 = plt.subplots(1)
for i in range(0,N):
    if y[i][0] > y[i][1]:
        plt.plot(x[i][0],x[i][1], "r*")
        rect = patches.Rectangle((x[i][0]-0.08, x[i][1]-0.08), 0.16, 0.16,edgecolor='r', facecolor="none")
        ax2.add_patch(rect)
    else:
        plt.plot(x[i][0],x[i][1], "b*")
        rect = patches.Rectangle((x[i][0]-0.08, x[i][1]-0.08), 0.16, 0.16,edgecolor='b', facecolor="none")
        ax2.add_patch(rect)
x_ax = np.linspace(0,1, 20)
y_ax = np.linspace(0,1, 20)
for i in range(0, 20):
    for j in range(0, 20):
        x_in = np.zeros((1,2))
        x_in[0][0]= x_ax[i]
        x_in[0][1] = y_ax[j]
        pred = model.predict(x_in)
        if pred[0][0] > pred[0][1]:
            plt.plot(x_in[0][0], x_in[0][1], 'y+')
        else:
            plt.plot(x_in[0][0], x_in[0][1], 'b+')

fig3 = plt.figure()
bounds = dual_network_bounds.dual_network_bounds(weights_t_n, depth, eps, x[0])
l = bounds[0]
u = bounds[1]
NN = 10000
NNN = 1000
inputs_x = x[0]+np.random.uniform(-eps,eps,(NN,2))
pred_outer_rob_1 = np.zeros((NN, 1))
pred_outer_rob_2 = np.zeros((NN, 1))
pred_pol_1 = np.zeros((NNN, 1))
pred_pol_2 = np.zeros((NNN, 1))
for i in range(0,NN):
    y = inputs_x[i]
    for j in range(0, depth):
        y = np.matmul(y, weights_t_n[2*j])+np.transpose(weights_t_n[2*j+1])
        for m in range(0, np.size(y, 1)):
            if l[j][0][m] < 0 and u[j][0][m]>0:
                if y[0][m] < 0:
                    y[0][m] = np.random.uniform(0, u[j][0][m]*(y[0][m]-l[j][0][m])/(u[j][0][m]-l[j][0][m]), 1)
                else:
                    y[0][m] = np.random.uniform(y[0][m], u[j][0][m]*(y[0][m]-l[j][0][m])/(u[j][0][m]-l[j][0][m]), 1)
            elif u[j][0][m] < 0:
                y[0][m] = 0
    y = np.matmul(y, weights_t_n[2*depth])+np.transpose(weights_t_n[2*depth+1])
    pred_outer_rob_1[i] = y[0][0]
    pred_outer_rob_2[i] = y[0][1]

for i in range(0, NNN):
    y = inputs_x[i]
    for j in range(0, depth):
        y = np.matmul(y, weights_t_n[2*j])+np.transpose(weights_t_n[2*j+1])
        for m in range(0, np.size(y, 1)):
            if y[0][m] < 0:
                y[0][m] = 0
    y = np.matmul(y, weights_t_n[2*depth])+np.transpose(weights_t_n[2*depth+1])
    pred_pol_1[i] = y[0][0]
    pred_pol_2[i] = y[0][1]

plt.plot(pred_outer_rob_1,pred_outer_rob_2 , 'r+')
plt.plot(pred_pol_1,pred_pol_2 , 'b+')

fig4 = plt.figure()
WW = model.get_weights()
bounds = dual_network_bounds.dual_network_bounds(WW, depth, eps, x[0])
l = bounds[0]
u = bounds[1]
NN = 10000
NNN = 1000
inputs_x = x[0]+np.random.uniform(-eps,eps,(NN,2))
pred_outer_rob_1 = np.zeros((NN, 1))
pred_outer_rob_2 = np.zeros((NN, 1))
pred_pol_1 = np.zeros((NNN, 1))
pred_pol_2 = np.zeros((NNN, 1))
for i in range(0,NN):
    y = inputs_x[i]
    for j in range(0, depth):
        y = np.matmul(y, WW[2*j])+np.transpose(WW[2*j+1])
        for m in range(0, np.size(y, 0)):
            if l[j][0][m] < 0 and u[j][0][m]>0:
                if y[m] < 0:
                    y[m] = np.random.uniform(0, u[j][0][m]*(y[m]-l[j][0][m])/(u[j][0][m]-l[j][0][m]), 1)
                else:
                    y[m] = np.random.uniform(y[m], u[j][0][m]*(y[m]-l[j][0][m])/(u[j][0][m]-l[j][0][m]), 1)
            elif u[j][0][m] < 0:
                y[m] = 0
    y = np.matmul(y, WW[2*depth])+np.transpose(WW[2*depth+1])
    pred_outer_rob_1[i] = y[0]
    pred_outer_rob_2[i] = y[1]


for i in range(0, NNN):
    y = inputs_x[i]
    for j in range(0, depth):
        y = np.matmul(y, WW[2*j])+np.transpose(WW[2*j+1])
        for m in range(0, np.size(y, 0)):
            if y[m] < 0:
                y[m] = 0
    y = np.matmul(y, WW[2*depth])+np.transpose(WW[2*depth+1])
    pred_pol_1[i] = y[0]
    pred_pol_2[i] = y[1]

plt.plot(pred_outer_rob_1,pred_outer_rob_2 , 'r+')
plt.plot(pred_pol_1,pred_pol_2 , 'b+')

plt.show()