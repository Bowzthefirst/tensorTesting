# -*- coding: utf-8 -*-
"""NumberGAN_Solution.ipynb




## **1) Importing Python Packages for GAN**
"""

import numpy as np
from keras.datasets import mnist
from keras.layers import BatchNormalization
from keras.layers import Dense, Reshape, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from os import mkdir
import os

#os.mkdir("generated_images")
os.mkdir("new_images")

os.mkdir("newImages")

"""## **2) Variables for Neural Networks & Data**"""

img_width = 28
img_height = 28
channels = 1
img_shape = (img_width, img_height, channels)
latent_dim = 100
adam = Adam(lr=0.0001)

"""## **3) Building Generator**




"""


def build_generator():
    model = Sequential()

    model.add(Dense(256, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Dense(np.prod(img_shape), activation='tanh'))
    model.add(Reshape(img_shape))

    model.summary()
    return model


generator = build_generator()

"""## **4) Building Discriminator**"""


def build_discriminator():
    model = Sequential()

    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()
    return model


discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

"""## **5) Connecting Neural Networks to build GAN**"""

GAN = Sequential()
discriminator.trainable = False
GAN.add(generator)
GAN.add(discriminator)

GAN.compile(loss='binary_crossentropy', optimizer=adam)
GAN.summary()

"""## **6) Outputting Images**

"""

# @title
## **7) Outputting Images**
import matplotlib.pyplot as plt
import glob
import imageio

save_name = 0.00000000


def save_imgs(epoch):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, latent_dim))
    gen_imgs = generator.predict(noise)
    global save_name
    save_name += 0.00000001
    print("%.8f" % save_name)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            # axs[i,j].imshow(gen_imgs[cnt])
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig("new_images/%.8f.png" % save_name)
    print('saved')
    plt.close()


"""## **7) Training GAN**"""


def train(epochs, batch_size=64, save_interval=200):
    (X_train, _), (_, _) = mnist.load_data()

    # print(X_train.shape)
    # Rescale data between -1 and 1
    X_train = X_train / 127.5 - 1.
    # X_train = np.expand_dims(X_train, axis=3)
    # print(X_train.shape)

    # Create our Y for our Neural Networks
    valid = np.ones((batch_size, 1))
    fakes = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        # Get Random Batch
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]

        # Generate Fake Images
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_imgs = generator.predict(noise)

        # Train discriminator
        d_loss_real = discriminator.train_on_batch(imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fakes)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (batch_size, latent_dim))

        # inverse y label
        g_loss = GAN.train_on_batch(noise, valid)

        print("******* %d [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

        if (epoch % save_interval) == 0:
            save_imgs(epoch)

    # print(valid)


train(30000, batch_size=64, save_interval=200)

"""### **8) Making GIF**"""

# Display a single image using the epoch number
# def display_image(epoch_no):
#   return PIL.Image.open('generated_images/%.8f.png'.format(epoch_no))

anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('generated_images/*.png')
    filenames = sorted(filenames)
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)

