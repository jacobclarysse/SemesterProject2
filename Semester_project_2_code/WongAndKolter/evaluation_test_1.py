import numpy as np
import matplotlib.pyplot as plt
import keras
import dual_network_bounds
import matplotlib.pyplot as plt


model = keras.models.load_model('test_model')
weights = model.get_weights()
x = np.random.uniform(0,1, (2,1))
depth = 2
eps = 0.1

bounds = dual_network_bounds.dual_network_bounds(weights, depth, eps, x)
u = bounds[1]
l = bounds[0]

N = 10000
adv = np.random.uniform(-eps, eps, (N, 2))

y_1 = np.zeros((N, 1))
y_2 = np.zeros((N, 1))
x_adv = np.zeros((N,2))
for i in range(0,N):
    x_adv[i][:] = adv[i][:]+np.transpose(x)

for i in range(0, N):
    e = x_adv[i][:]
    for j in range(0, depth):
        e = np.matmul(e, weights[2*j])+weights[2*j+1]
        u_j = u[j][0]
        l_j = l[j][0]
        for m in range(0, np.size(e,0)):
            if u_j[m] > 0 and l_j[m]<0 and e[m] < 0:
                e[m] = np.random.uniform(0, (u_j[m]*e[m]-l_j[m]*u_j[m])/(u_j[m]-l_j[m]))
            elif u_j[m] > 0 and l_j[m]<0 and e[m] >= 0:
                e[m] = np.random.uniform(e[m], (u_j[m]*e[m]-l_j[m]*u_j[m])/(u_j[m]-l_j[m]))
            elif u_j[m] < 0:
                e[m] = 0

    e = np.matmul(e, weights[2*depth]) +weights[2*depth+1]
    #e = e[0]
    y_1[i] = e[0]
    y_2[i] = e[1]

#print(weights)
plt.figure(1)
plt.plot(y_1, y_2, 'r*')
y = np.transpose((model.predict(x_adv[0:1000])))
plt.plot(y[0][:], y[1][:], 'b*')
plt.show()