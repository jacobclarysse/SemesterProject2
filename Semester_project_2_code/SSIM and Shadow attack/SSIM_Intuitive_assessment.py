import numpy as np
import matplotlib.pyplot as plt
import pathlib
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras import layers
import os


batch_size = 32
img_height = 180
img_width = 180
sigma = 0.25
N = 10
lambda_tv = 10
lambda_s = 350
lambda_c = 100

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file(origin=dataset_url, fname='flower_photos', untar=True)
data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

roses = list(data_dir.glob('roses/*'))
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

for images, labels in train_ds.take(1):
    for i in range(9):
        #ax = plt.subplot(3, 3, i + 1)
        #plt.imshow(images[i].numpy().astype("uint8"))
        if i == 0:
            image_s = images[i].numpy().astype("uint8")
            image = np.reshape(images[i].numpy(),(180,180,3))/255.0
        #plt.title(class_names[labels[i]])
        #plt.axis("off")

def rgb2gray(rgb):
    r = rgb[:,:, 0]
    g = rgb[:,:, 1]
    b = rgb[:,:, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

x = image
y = x
#y = (np.mean(y)+(y-np.mean(y))/16).astype('uint8')
#y  = y/255.0
y = y + np.random.normal(0,1.0, (180,180, 3))
#for i in range(10, np.size(y,0)-10):
#    for j in range(10, np.size(y,1)-10):
#        y[i,j] = np.mean(y[i-5:i+5,j-5:j+5])
def luminance(x,y):
    C1 = (0.01)**2
    lum = (2*np.mean(x)*np.mean(y)+C1)/(np.mean(x)**2+np.mean(y)**2+C1)
    return lum

def contrast(x,y):
    C2 = (0.03)**2
    std_dev_x = np.sqrt(np.sum((x-np.mean(x))**2)/(180*180-1))
    #print(np.sqrt(np.var(x)))
    #print(std_dev_x)
    std_dev_y = np.sqrt(np.sum((y-np.mean(y))**2)/(180*180-1))
    #print(std_dev_y)
    contr = (2*std_dev_x*std_dev_y+C2)/(std_dev_x**2+std_dev_y**2+C2)
    return contr

def structural_sim(x,y):
    C3 = ((0.03)**2)/2
    std_dev_x = np.sqrt(np.sum((x-np.mean(x))**2)/(180*180-1))
    std_dev_y = np.sqrt(np.sum((y-np.mean(y))**2)/(180*180-1))
    cor = np.sum(np.multiply((y-np.mean(y)),(x-np.mean(x))))/(180*180-1)
    s = (cor+C3)/(std_dev_x*std_dev_y+C3)
    return s

#print(luminance(x,y))
#print(contrast(x,y))
#print(structural_sim(x,y))
plt.figure(1)
plt.imshow(x)
plt.figure(2)
plt.imshow(y)
plt.show()