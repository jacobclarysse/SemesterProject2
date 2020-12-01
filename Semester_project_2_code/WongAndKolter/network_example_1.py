import numpy as np
import matplotlib.pyplot as plt
import keras
import keras.layers as ly

###Training example: start grid of 4 
x= np.random.uniform(0, 1, (10000,2))
y = np.zeros((10000, 2))
for i in range(0, 10000):
    if x[i][0]*x[i][0]+x[i][1]*x[i][0]>= 0.25 and x[i][0]*x[i][0]+x[i][1]*x[i][0] < 0.8*0.8:
        y[i][0] = 1
    else:
        y[i][1] = 1

###Define model
input = ly.Input(shape = (2,))
l1 = ly.Dense(10, activation= 'relu')(input)
l2 = ly.Dense(10, activation='relu')(l1)
output = ly.Dense(2, activation='relu')(l2)
model = keras.Model(inputs = input, outputs= output)

model.summary()
model.compile(optimizer = "sgd", loss="binary_crossentropy")
model.fit(x,y, epochs=5)

model.save('test_model')