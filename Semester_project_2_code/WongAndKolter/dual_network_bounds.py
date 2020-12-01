import numpy as np
import matplotlib.pyplot as plt
import keras

def absolute_of_matrix(W):
    res = np.zeros((np.size(W, 1),1))
    for i in range(0, np.size(W, 1)):
        res[i] = np.linalg.norm(W[:, i], 1)
    return np.transpose(res)

def portion(v):
    neg = (v-np.abs(v))/2
    pos = (v+np.abs(v))/2
    port = [neg, pos]
    return port

def form_D_I(l, u):
    S = np.size(l,1)
    l = l[0]
    u = u[0]
    D_I = np.eye(S)
    for i in range(0,S):
        if u[i] < 0:
            D_I[i][i] = 0
        elif l[i] < 0:
            D_I[i][i] = u[i]/(u[i]-l[i])
    return D_I 


##Direct implementation first
def dual_network_bounds(weights, depth, eps, x):
    nu_hat = [weights[0]]
    la = [np.transpose(weights[1])]
    l = []
    u = []
    D = []
    nu = []
    x_tr = np.transpose(x)
    absW1 = absolute_of_matrix(nu_hat[0])
    l.append(np.matmul(x_tr, nu_hat[0])+la[0]-eps*absW1)
    u.append(np.matmul(x_tr, nu_hat[0])+la[0]+eps*absW1)
    for i in range(2, depth+1):
        DI = form_D_I(l[i-2], u[i-2])
        D.append(DI)
        #Initialise new terms
        nu_i = np.matmul(DI, weights[2*(i-1)])
        la_i = weights[2*(i-1)+1]
        la.append(np.transpose(la_i))
        nu.append(nu_i)
        #Propagate existing terms
        D_IWI = np.matmul(DI,weights[2*(i-1)])
        for j in range(2, i):
            nu[j-2]= np.matmul(nu[j-2], D_IWI)
        
        for j in range(1,i):
            la[j-1] = np.matmul(la[j-1],D_IWI)
        nu_hat[0] = np.matmul(nu_hat[0], D_IWI)
        #compute bounds
        lam = 0
        for j in la:
            lam = lam + j
        ab = absolute_of_matrix(nu_hat[0])
        #for j in range(0, np.size(nu_hat[0],0)):
         #   ab = ab + np.absolute(nu_hat[0][j])
        psi_i = np.matmul(x_tr, nu_hat[0]) + lam
        l_l = 0
        l_u = 0
        for m in range(2, i+1):
            l_m = l[m-2]
            l_m = l_m[0]
            u_m = u[m-2]
            u_m = u_m[0]
            for q in range(0, np.size(l_m)):
                if l_m[q] < 0 and u_m[q]>0:
                    portions = portion(nu[m-2][q])
                    l_l = l_l - l_m[q]*np.transpose(portions[0])
                    l_u = l_u - l_m[q]*np.transpose(portions[1])
        l.append(psi_i-eps*ab+l_l)
        u.append(psi_i+eps*ab-l_u)
        b = [l,u]
    return b

