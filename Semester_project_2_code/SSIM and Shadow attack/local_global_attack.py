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
SSIM_max = 0.9
batch_size = 32
img_height = 180
img_width = 180
L = 10
M = img_height*img_width/(L**2)
lambda_1 = 1
lambda_2 = 0.1*np.ones((100,1))
sigma = 0.1
steps = 100
alpha = np.sqrt(SSIM_max)
eta = np.sqrt(SSIM_max)
N = 100
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
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
model.summary()

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
print(prediction)
print(class_names[np.argmax(prediction)])
image[0] = image[0]/255.0
im_gray = SSIM_utils.rgb2gray(image[0])
mean_min, mean_max = SSIM_utils.max_mean(im_gray, alpha)
var_m = SSIM_utils.max_var(im_gray, eta)
initial_val = np.random.normal((mean_min+mean_max)/2,var_m, (180,180))
delta = tf.Variable(initial_value=np.reshape(initial_val,(180,180,1)).astype("float32"), trainable = True)
weights = [delta]


bce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM, from_logits=True)
def loss(original, d):
    delta_rgb_tr =  tf.image.grayscale_to_rgb(d)
    y_s = tf.random.normal([N, 180,180,3], original+delta_rgb_tr, sigma, tf.float32)
    y_pred = tf.transpose(model(y_s*255))
    y_true = tf.convert_to_tensor(np.tile(np.array([[1,0,0,0,0]]).transpose(), (1, N)).astype("float32"))
    loss_v = 0
    for i in range(0,int(img_height/L)):
        for j in range(0,int(img_width/L)):
            loss_v = loss_v + lambda_2[i+j]*tf.math.reduce_variance(d[i:i+L,j+L])
    loss_v = loss_v + bce(y_pred, y_true)/N+lambda_1*tf.math.abs(tf.math.reduce_mean(d))
    return loss_v

def grad(img, weights):
    with tf.GradientTape() as tape:
        loss_value = loss(img, weights[0])
    return loss_value, tape.gradient(loss_value, weights)

image_t = tf.convert_to_tensor(image[0].astype("float32"))
image_gray = SSIM_utils.rgb2gray(image[0])
for i in range(steps):
    if i%10 == 0:
        plt.figure(i/2+3)
        delta_rgb = tf.image.grayscale_to_rgb(weights[0])
        delta_p = (delta_rgb.numpy()*255).astype("int32")
        img_spoof = np.clip(image_s+delta_p, 0, 255)
        print("l2-norm: "+str(np.linalg.norm(delta_p/255.0)))
        print("l_infty norm: "+str(np.amax(delta_p)))
        print("SSIM: "+str(SSIM_utils.SSIM(image_gray, SSIM_utils.rgb2gray(img_spoof.astype('float32')/255.0))))
        plt.imshow(img_spoof)
        print(i)
    loss_value, grads = grad(image_t, weights)
    optimizer.apply_gradients(zip(grads, weights))
    if tf.math.reduce_mean(weights[0]) > mean_max or tf.math.reduce_mean(weights[0]) < mean_min:
        lambda_1 = 2*lambda_1
        print("need bigger mean! "+str(tf.math.reduce_mean(weights[0])))
    if tf.math.reduce_variance(weights[0]) > var_m:
        lambda_2 = 2*lambda_2
        print("need bigger var! "+str(tf.math.reduce_variance(weights[0])))

weights[0] = tf.image.grayscale_to_rgb(weights[0])
image_new = [np.clip((image[0]+weights[0].numpy())*255, 0, 255)]
print(model.predict(np.reshape(image_new, (1,180,180,3))))
print(class_names[np.argmax(model.predict(np.reshape(image_new, (1,180,180,3))))])
plt.figure(100)
perturbation_img = np.clip(255*weights[0].numpy()+np.absolute(np.min(weights[0].numpy())*255),0,255).astype('uint8')
plt.imshow(perturbation_img)
plt.show()
