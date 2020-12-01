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
k_B = 4
k_D = 2
k_B_l = [0,0,0,0]
k_D_l = [0,0]
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
gamma_B = 1.5
gamma_D = 1.5
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

@tf.function
def special_norm_matrice(matrix, S):
    #beg = timer()
    v = tf.reshape(matrix,(S, 28,28))
    m_e = 0.0
    for i in range(4):
        for j in range(4):
            m_n = tf.norm(v[:,i*7:i*7+7,j*7:j*7+7], ord=2, axis=[-1,-2])**2
            m_e = tf.maximum(m_e, m_n)
    #en = timer()
    #print("time to norm "+str(en-beg))
    return m_e

@tf.function
def form_D(outputs):
    #S = 100
    #D_I = tf.Variable(initial_value = tf.eye(S), trainable = False)
    D = tf.reshape((tf.math.sign(outputs)+1)/2,(1,100))
    #for i in range(0,S):
    #    if outputs[i] < 0:
    #       D_I[i,i].assign(0)
    return tf.reshape(tf.linalg.diag(D),(100,100))

@tf.function
def linear_layer(W1,x_in, b):
    return W1@x_in+b

@tf.function
def mult(a,b):
    return a@b

@tf.function
def distances(l,M,norms):
    #abs_l = tf.reshape(tf.math.abs(l), (1,100))
    abs_l = tf.math.abs(l)
    dist = tf.divide(abs_l[0], tf.reshape(M*norms+0.0001, (1,100)))
    return dist[0]

@tf.function
def signed_distances(l,M,norms):
    dist = tf.divide(l[0], tf.reshape(M*norms+0.0001, (1,10)))
    return dist[0]

##Start with 2 layer inefficient
def MMR(input, weights_t, target,norms):
    input = tf.reshape(input,(784,1))
    #l1 = weights_t[0]@input+weights_t[1]
    l1 = linear_layer(weights_t[0], input, weights_t[1])
    #l1_abs = tf.math.abs(l1)
    dist = distances(l1 ,M, norms)
    val, ind = tf.math.top_k(-dist, k=4, sorted=True, name=None)
    for i in ind:
        #dist = l1_abs[i]/(M*norms[i])
        i_min = k_B_l.index(min(k_B_l)) ##should via heap
        if dist[i]>k_B_l[i_min]:
            k_B_l[i_min] = dist[i]
        else:
            break
    D = form_D(l1)
    #print("shape of D: "+str(tf.shape(D)))
    #V2 = weights_t[2]@D@weights_t[0]
    V2_s = mult(weights_t[2], D)
    V2 = mult(V2_s, weights_t[0])
    #a_2 = weights_t[2]@D@weights_t[1]+weights_t[3]
    a_2 = linear_layer(V2_s, weights_t[1], weights_t[3])
    #l2 = V2@input+a_2
    l2 = linear_layer(V2, input, a_2)  
    #l2_abs = tf.math.abs(l2)
    norms_2 = special_norm_matrice(V2, 100)
    dist = distances(l2, M, norms_2)
    val, ind = tf.math.top_k(-dist, k=4, sorted=True, name=None)
    for i in ind:
        #dist = l2_abs[i]/(M*norms_2[i])
        i_min = k_B_l.index(min(k_B_l)) ##should via heap
        if dist[i]>k_B_l[i_min]:
            k_B_l[i_min] = dist[i]
        else:
            break
    D2 = form_D(l2)
    #V3 = weights_t[4]@D2@V2
    V3_s = mult(weights_t[4], D2)
    V3 = mult(V3_s, V2)
    #a_3 = weights_t[4]@D2@a_2+weights_t[5]
    a_3 = linear_layer(V3_s, a_2, weights_t[5])
    i_t = tf.math.argmax(target)
    #output = V3@input+a_3
    output = linear_layer(V3, input, a_3)
    out = output[i_t]-output
    V_c_br = tf.reshape(V3[i_t,:],(1,784))-V3
    norms_3 = special_norm_matrice(V_c_br, 10) 
    dist = signed_distances(out, M, norms_3)
    val, ind = tf.math.top_k(-dist, k=2, sorted=True, name=None)
    for i in ind:
        i_min = k_D_l.index(min(k_D_l)) ##should via heap
        if dist[i]>k_D_l[i_min]:
            k_D_l[i_min] = dist[i]
        else:
            break
    loss_k_B = 0
    for i in range(k_B):
        l = 1-k_B_l[i]/gamma_B
        k_B_l[i]=0
        if l < 0:
            continue
        loss_k_B = loss_k_B + l
    loss_k_D = 0
    for i in range(k_D):
        l = 1-k_D_l[i]/gamma_D
        k_D_l[i]=0
        if l<0:
            continue
        loss_k_D = loss_k_D + l
    return loss_k_D/k_D+loss_k_B/k_B+ce(tf.reshape(target,(1,10)),tf.reshape(output,(1,10)))

def grad(inputs, targets):
    Batch_Size = np.size(targets, 0)
    #input_t = tf.convert_to_tensor(inputs.astype('float32'))
    #targets_t = tf.convert_to_tensor(targets.astype('float32'))
    with tf.GradientTape() as tape:
        loss_value = 0
        #predictions = predictions_t(input_t, weights_t)
        MMR_v=0
        norms = special_norm_matrice(weights_t[0], 100)
        for i in range(0, Batch_Size):
            MMR_v_s= MMR(inputs[i,:],weights_t,targets[i,:],norms)
            MMR_v += MMR_v_s 
        #loss_value = (ce(tf.transpose(targets_t), predictions)+MMR_v)/Batch_Size
        loss_value = MMR_v/Batch_Size
    return loss_value, tape.gradient(loss_value, weights_t)

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
        loss_value, grads = grad(x_train[batches_size*i:batches_size*i+batches_size][:], y[batches_size*i:batches_size*i+batches_size][:])
        optimizer.apply_gradients(zip(grads, weights_t))
        #end = timer()
        #print("time for batch: "+str(end-begin))
        mean_loss = mean_loss+loss_value/batches
        p = p+1
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

y_pred = predictions_t(x_test, weights_t)
score = 0
for i in range(10000):
    if tf.math.argmax(y_pred[:,i]) == Y_test[i]:
        score = score+1
print("test accuracy: "+str(score/10000))
for i in range(len(weights)):
    weights[i] = weights_t[i].numpy()
np.save('weights_test_5.npy', weights)
