import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import pathlib
import tensorflow_datasets as tfds
from tensorflow.keras import layers
import os
import SSIM_utils


###Constants
SSIM_max = 0.99
batch_size = 32
img_height = 180
img_width = 180
alpha = np.sqrt(SSIM_max)
eta = np.sqrt(SSIM_max)

###Import flower database as always + show first picture as example

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

model = tf.keras.models.load_model('flower_model_random_gaussian_noise_augmentation.h5')
#model.summary()
for images, labels in train_ds.take(1):
    for i in range(9):
        #ax = plt.subplot(3, 3, i + 1)
        #plt.imshow(images[i].numpy().astype("uint8"))
        if i == 0:
            image_s = images[i].numpy().astype("uint8")
            image = [np.reshape(images[i].numpy().astype("float32"),(180,180,3))]
        #plt.title(class_names[labels[i]])
        #plt.axis("off")

plt.figure(1)
plt.imshow(image_s)
prediction = model.predict(np.reshape(image, (1,180,180,3)))
#print(prediction)
#print(class_names[np.argmax(prediction)])
image[0] = image[0]/255.0

im_gray = SSIM_utils.rgb2gray(image[0])
mean_min, mean_max = SSIM_utils.max_mean(im_gray, alpha)
print("found max mean: "+str(mean_max))
print("found min mean: "+str(mean_min))
var_m = SSIM_utils.max_var(im_gray, eta)
print("found max var:"+str(var_m))
std_var_min = SSIM_utils.low_std_dev(im_gray, 0.99)
print("found min std_dev: "+str(std_var_min))
print("found outer l2 ball of: "+str(std_var_min*np.sqrt(180*180-1)))
print("found outer l2 ball of: "+str(np.sqrt((180*180-1)*var_m)))

delta_ex = np.random.normal(mean_max, np.sqrt(var_m), (img_width, img_height))
print("max infinity found max mean: "+str(np.max(np.absolute(delta_ex))))
print("l_2 norm max mean: "+str(np.linalg.norm(delta_ex)))
pertubation = np.clip(im_gray+delta_ex, 0, 1)

delta_ex_2 = np.random.normal(mean_min, np.sqrt(var_m), (img_width, img_height))
print("max infinity found min mean: "+str(np.max(np.absolute(delta_ex_2))))
print("l_2 norm min mean: "+str(np.linalg.norm(delta_ex_2)))
pertubation2 = np.clip(im_gray+delta_ex_2, 0, 1)

SSIM = SSIM_utils.SSIM(im_gray, pertubation)
print("True SSIM pert_max: "+str(SSIM))
SSIM_2 = SSIM_utils.SSIM(im_gray, pertubation2)
print("True SSIM pert_min: "+str(SSIM_2))

plt.figure(2)
plt.imshow(im_gray, cmap="gray")
plt.figure(3)
plt.imshow(pertubation, cmap="gray")
plt.figure(4)
plt.imshow(pertubation2, cmap="gray")

rgb_pert = SSIM_utils.to_rgb(delta_ex, image[0])
rgb_pert2 = SSIM_utils.to_rgb(delta_ex_2, image[0])
plt.figure(5)
plt.imshow(rgb_pert, cmap=None)
plt.figure(6)
plt.imshow(rgb_pert, cmap=None)
plt.show()
