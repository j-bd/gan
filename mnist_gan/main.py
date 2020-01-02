#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 16:00:04 2019

@author: latitude
"""
import numpy as np
from keras.layers import Dense, Dropout, Input
from keras.models import Model,Sequential
from keras.datasets import mnist
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import adam


class Gan():


def adam_optimizer():
    return adam(lr=0.0002, beta_1=0.5)

def discriminator():
    discriminator=Sequential()
    discriminator.add(Dense(units=1024,input_dim=784))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Dense(units=512))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Dense(units=256))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dense(units=1, activation='sigmoid'))
    discriminator.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
    return discriminator

def generator():
    generator=Sequential()
    generator.add(Dense(units=256,input_dim=100))
    generator.add(LeakyReLU(0.2))
    generator.add(Dense(units=512))
    generator.add(LeakyReLU(0.2))
    generator.add(Dense(units=1024))
    generator.add(LeakyReLU(0.2))
    generator.add(Dense(units=784, activation='tanh'))
    generator.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
    return generator

# define a composite gan model for the generator and discriminator
def define_gan(generator, discriminator):
    # make weights in the discriminator not trainable
    discriminator.trainable = False
    # connect them
    model = Sequential()
    # add generator
    model.add(generator)
    # add the discriminator
    model.add(discriminator)
    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

# gan training algorithm
def train_gan(generator, discriminator, dataset, latent_dim, n_epochs, n_batch):
    # calculate the number of batches per epoch
    batches_per_epoch = int(len(dataset) / n_batch)
    # calculate the number of training iterations
    n_steps = batches_per_epoch * n_epochs
    # gan training algorithm
    for i in range(n_steps):
        # generate points in the latent space
        z = np.random.randn(latent_dim * n_batch)
        # reshape into a batch of inputs for the network
        z = z.reshape(n_batch, latent_dim)
        # generate fake images
        X_fake = generator.predict(z)
        # select a batch of random real images
        ix = np.random.randint(0, len(dataset), n_batch)
        # retrieve real images
        X_real = dataset[ix]
        # update weights of the discriminator model
        # ...
        # define target labels for fake images
        y_fake = np.zeros((n_batch, 1))
        # update the discriminator for fake images
        discriminator.train_on_batch(X_fake, y_fake)
        # define target labels for real images
        y_real = np.ones((n_batch, 1))
        # update the discriminator for real images
        discriminator.train_on_batch(X_real, y_real)
        #
        #
#        # generate points in the latent space
#        z = np.random.randn(latent_dim * n_batch)
#        # reshape into a batch of inputs for the network
#        z = z.reshape(n_batch, latent_dim)
#        # generate fake images
#        fake = generator.predict(z)
#        # classify as real or fake
#        result = discriminator.predict(fake)
#        # update weights of the generator model
#        # ...
        # generate points in the latent space
        z = np.random.randn(latent_dim * n_batch)
        # reshape into a batch of inputs for the network
        z = z.reshape(n_batch, latent_dim)
        # define target labels for real images
        y_real = np.ones((n_batch, 1))
        # update generator model
        gan_model.train_on_batch(z, y_real)
