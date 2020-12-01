from sklearn.linear_model import LogisticRegression
import numpy as np
import tensorflow as tf

p = 100
theta_true = 1/np.sqrt(p)*np.ones((p,1))
eps = 0.05
N = int(10*p/(eps**2))
x = np.random.normal(0,1,(N,p))
y_true = np.sign(np.matmul(x, theta_true))
y_training = y_true
errors = int(eps*N)
i = 0
while errors != 0:
    if y_true[i] == 1:
        x[i,:] = p**2*x[i,:]
        y_training[i] = -1
        errors = errors-1
    i = i + 1
#y_training = tf.keras.utils.to_categorical(y_training)
clf = LogisticRegression(random_state=0).fit(x, y_training)
N_test = 1000
x_clean = np.random.normal(0,1,(N_test,p))
correct = 0
pred = clf.predict_proba(x_clean)
y_true = np.sign(np.matmul(x_clean, theta_true))
print(y_true[0])
print(pred[0])
print(np.argmax(pred[0]))
for i in range(N_test):
    if np.argmax(pred[i]) == int((y_true[i]+1)/2):
        correct = correct+1

print(correct/N_test)