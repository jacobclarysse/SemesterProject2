import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras import layers
#from tensorflow.keras.datasets import mnist
from timeit import default_timer as timer

def relu(X):
    return np.maximum(0,X)

def predict(input, weights):
    l1 = relu(np.matmul(input, np.transpose(weights[0]))+np.transpose(weights[1]))
    l2 = relu(np.matmul(l1,np.transpose(weights[2]))+np.transpose(weights[3]))
    l3 = np.matmul(l2, np.transpose(weights[4]))+np.transpose(weights[5])
    return l3

def predict_t(input, weights):
    input = tf.reshape(input, (1,784))
    l1 = tf.nn.relu(input @ tf.transpose(weights[0])+tf.transpose(weights[1]))
    l2 = tf.nn.relu(l1 @ tf.transpose(weights[2])+tf.transpose(weights[3]))
    l3 = l2 @ tf.transpose(weights[4])+tf.transpose(weights[5])
    return l3

def special_norm_matrice(matrix, S):
    #beg = timer()
    v = np.reshape(matrix,(S, 28,28))
    m_e = 0
    for i in range(4):
        for j in range(4):
            m_n = np.linalg.norm(v[:,i*7:i*7+7,j*7:j*7+7], ord=2, axis=(-1,-2))
            m_e = m_e+m_n#np.maximum(m_e, m_n)
    #en = timer()
    #print("time to norm "+str(en-beg))
    return m_e

def form_D(outputs):
    D = np.reshape((np.sign(outputs)+1)/2,(100,))
    return np.reshape(np.diag(D), (100,100))

def linear_layer(W1,x_in, b):
    return np.matmul(W1, x_in)+b

def mult(a,b):
    return np.matmul(a, b)

def distances(l,M,norms):
    #abs_l = tf.reshape(tf.math.abs(l), (1,100))
    abs_l = np.absolute(l)
    dist = np.divide(abs_l[0], np.reshape(norms, (1,100)))
    return dist[0]

def signed_distances(l,M,norms):
    dist = np.divide(np.reshape(l,(1,10)), np.reshape(0.0001+norms, (1,10)))
    return dist[0]

def best_certificate(x_in, norms, target, weights, M):
    x_in = np.reshape(x_in,(784,1))
    #l1 = weights_t[0]@input+weights_t[1]
    l1 = linear_layer(weights[0], x_in, weights[1])
    #l1_abs = tf.math.abs(l1)
    dist = distances(l1 ,M, norms)
    val_1 = np.min(dist)
    D = form_D(l1)

    V2_s = mult(weights[2], D)
    V2 = mult(V2_s, weights[0])
    a_2 = linear_layer(V2_s, weights[1], weights[3])
    l2 = linear_layer(V2, x_in, a_2)  
    norms_2 = special_norm_matrice(V2, 100)
    dist = distances(l2, M, norms_2)
    val_2 = np.min(dist)
    val = np.minimum(val_1,val_2)
    D2 = form_D(l2)

    V3_s = mult(weights[4], D2)
    V3 = mult(V3_s, V2)
    a_3 = linear_layer(V3_s, a_2, weights[5])
    i_t = tf.math.argmax(target)
    output = linear_layer(V3, x_in, a_3)
    out = output[i_t]-output
    V_c_br = tf.reshape(V3[i_t,:],(1,784))-V3
    norms_3 = special_norm_matrice(V_c_br, 10) 
    dist = signed_distances(out, M, norms_3)
    dist[i_t] += 10
    val_3 = np.min(dist)
    val = np.minimum(val,val_3)
    return val

###Gray version
def TV(delta):
    x_wise = delta[:,1:]-delta[:,:-1]
    y_wise = delta[1:,:]-delta[:-1,:]
    tvv = tf.reduce_sum(x_wise*x_wise)+tf.reduce_sum(y_wise*y_wise)
    return tvv

def c(delta):
    m = tf.math.reduce_mean(tf.abs(delta))
    return m
