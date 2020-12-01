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
from kerassurgeon.operations import delete_layer

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

optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
image_t = tf.convert_to_tensor(image[0].astype("float32"))
initial_val = np.random.normal(0,0.1, (180,180,3))
#init = np.zeros((180,180,3))
#init[:,:,0] = initial_val
#init[:,:,1] = initial_val
#init[:,:,2] = initial_val
no_dissim = False
#delta = tf.Variable(initial_value=init.astype("float32"), trainable = True)
delta = tf.Variable(initial_value=np.reshape(initial_val,(180,180,3)).astype("float32"), trainable = True)
steps = 100
weights = [delta]
cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM, from_logits=True)

def TV(delta):
    x_wise = delta[:,1:,:]-delta[:,:-1,:]
    y_wise = delta[1:,:,:]-delta[:-1,:,:]
    tvv = tf.reduce_sum(x_wise*x_wise)+tf.reduce_sum(y_wise*y_wise)
    return tvv

def c(delta):
    m = tf.convert_to_tensor([tf.math.reduce_mean(tf.abs(delta[:,:,0])), tf.math.reduce_mean(tf.abs(delta[:,:,1])), tf.math.reduce_mean(tf.abs(delta[:,:,2]))])
    #print(tf.reduce_mean(tf.abs(delta)))
    ccc = tf.norm(m)**2
    return ccc
def dissim(delta):
    m = tf.convert_to_tensor([(delta[:,:,0]-delta[:,:,1])**2, (delta[:,:,0]-delta[:,:,2])**2, (delta[:,:,1]-delta[:,:,2])**2])
    d = tf.norm(m)**2
    return d

def loss(img, delta):
    if no_dissim:
        delta =  tf.image.grayscale_to_rgb(delta)
        y_s = tf.random.normal([N, 180,180,3], img+delta, sigma, tf.float32)
    else:
        y_s = tf.random.normal([N, 180,180,3], img+delta, sigma, tf.float32)
    y_pred = tf.transpose(model(y_s*255))
    y_true = tf.convert_to_tensor(np.tile(np.array([[0,0,0,1,0]]).transpose(), (1, N)))
    loss_v = cce(y_true, y_pred)+lambda_tv*TV(delta)+lambda_c*c(delta)+lambda_s*dissim(delta)
    return loss_v

def grad(img, weights):
    with tf.GradientTape() as tape:
        loss_value = loss(img, weights[0])
    return loss_value, tape.gradient(loss_value, weights)

for i in range(steps):
    if i%10 == 0:
        plt.figure(i/10+3)
        if no_dissim:
            delta_rgb = tf.image.grayscale_to_rgb(delta)
            delta_p = (delta_rgb.numpy()*255).astype("int32")
        else:
            delta_p = (delta.numpy()*255).astype("int32")
        img_spoof = np.clip(image_s+delta_p, 0, 255)
        print("l2-norm: "+str(np.linalg.norm(delta_p/255.0)))
        print("l_infty norm: "+str(np.amax(delta_p)))
        plt.imshow(img_spoof)
        print(i)
    loss_value, grads = grad(image_t, weights)
    optimizer.apply_gradients(zip(grads, weights))
if no_dissim:
    weights[0] = tf.image.grayscale_to_rgb(weights[0])
image_new = [np.clip((image[0]+weights[0].numpy())*255, 0, 255)]
print(model.predict(np.reshape(image_new, (1,180,180,3))))
print(class_names[np.argmax(model.predict(np.reshape(image_new, (1,180,180,3))))])
plt.figure(100)
perturbation_img = np.clip(255*weights[0].numpy()+np.absolute(np.min(weights[0].numpy())*255),0,255).astype('uint8')
plt.imshow(perturbation_img)
plt.show()