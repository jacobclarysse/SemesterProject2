import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

plt.figure(1)
plt.imshow(train_images[0])
pert = np.zeros((32,32,3))
for i in range(32):
    for j in range(32):
        for m in range(0,3):
            if i<5 or i>13 or j>25 or j<15:
                pert[i,j,m]=0
            elif train_images[0][i,j,m] < 0.9:
                pert[i,j,m]=pert[i,j,m]+np.random.uniform(0,0.22)
print(np.linalg.norm(pert))
plt.figure(2)
perturbed_im = train_images[0] + pert
plt.imshow(perturbed_im)
plt.show()