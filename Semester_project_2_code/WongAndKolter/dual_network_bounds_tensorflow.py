import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def absolute_of_matrix_t(W):
    S_W = tf.shape(W)
    res = tf.Variable(initial_value = tf.zeros((S_W[1],1)), trainable = False)
    for i in range(0, S_W[1]):
        res[i].assign(tf.reshape(tf.norm(W[:,i], 1),(1,)))
    return tf.transpose(res)

def portion_t(v):
    neg = (v-tf.abs(v))/2
    pos = (v+tf.abs(v))/2
    port = [neg, pos]
    return port

def form_D_I_t(l, u):
    S = tf.shape(l)[1]
    l = l[0]
    u = u[0]
    D_I = tf.Variable(initial_value = tf.eye(S), trainable = False)
    for i in range(0,S):
        if u[i] < 0:
            D_I[i,i].assign(0)
        elif l[i] < 0:
            D_I[i,i].assign(u[i]/(u[i]-l[i]))
    return D_I 

def form_D_I_t_2(l, u):
    S = tf.shape(l)[1]
    l = l[0]
    u = u[0]
    D_I = tf.Variable(initial_value = tf.eye(S), trainable = False)
    indexes = []
    for i in range(0,S):
        if u[i] < 0:
            D_I[i,i].assign(0)
        elif l[i] < 0:
            D_I[i,i].assign(u[i]/(u[i]-l[i]))
            indexes.append(i)
    return D_I, indexes 


##Direct implementation first
def dual_network_bounds_t(weights, depth, eps, x):
    x = tf.reshape(x, (2,1))
    nu_hat = [weights[0]]
    la = [tf.transpose(weights[1])]
    l = []
    u = []
    nu = []
    x_tr = tf.transpose(x)
    absW1 = absolute_of_matrix_t(nu_hat[0])
    l.append(x_tr@nu_hat[0]+la[0]-eps*absW1)
    u.append(x_tr@nu_hat[0]+la[0]+eps*absW1)
    for i in range(2, depth+1):
        DI, indexes = form_D_I_t_2(l[i-2], u[i-2])
        #Initialise new terms
        nu_i = DI @ weights[2*(i-1)]
        la_i = weights[2*(i-1)+1]
        la.append(tf.transpose(la_i))
        nu.append(nu_i)
        #Propagate existing terms
        D_IWI = DI @ weights[2*(i-1)]
        for j in range(2, i):
            nu[j-2]= nu[j-2] @ D_IWI
        
        for j in range(1,i):
            la[j-1] = la[j-1] @ D_IWI
        nu_hat[0] = nu_hat[0] @ D_IWI
        #compute bounds
        lam = tf.Variable(initial_value = tf.zeros((1, tf.shape(la[0])[1])), trainable = False)
        for j in la:
            lam.assign(lam + j)
        ab = absolute_of_matrix_t(nu_hat[0])
        #for j in range(0, np.size(nu_hat[0],0)):
         #   ab = ab + np.absolute(nu_hat[0][j])
        psi_i = x_tr @ nu_hat[0] + lam
        l_l = tf.Variable(initial_value = tf.zeros((1, tf.shape(la[0])[1])), trainable = False)
        l_u = tf.Variable(initial_value = tf.zeros((1, tf.shape(la[0])[1])), trainable = False)
        for m in range(2, i+1):
            l_m = l[m-2]
            l_m = l_m[0]
            u_m = u[m-2]
            u_m = u_m[0]
            for q in indexes:
                portions = portion_t(nu[m-2][q])
                l_l.assign(l_l - l_m[q]*tf.transpose(portions[0]))
                l_u.assign(l_u - l_m[q]*tf.transpose(portions[1]))
        l.append(psi_i-eps*ab+l_l)
        u.append(psi_i+eps*ab-l_u)
        bounds = [l,u]
    return bounds

