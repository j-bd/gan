#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 11:19:52 2020

@author: j-bd
"""

import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import adam


class Gan:
    '''gan implementation'''
    def init(self, dataset_size, latent_dim, n_epochs, n_batch):
        '''Instance a new gan object'''
        self.latent_dim = latent_dim
        self.n_batch = n_batch
        self.size = dataset_size

        self.batches_per_epoch = int(self.size / n_batch)
        self.n_steps = self.batches_per_epoch * n_epochs

        self.generator = self.define_generator()
        self.discriminator = self.define_discriminator()
        self.gan = self.define_gan(self.generator, self.discriminator)

    def adam_optimizer(self):
        '''Define optimizer type'''
        return adam(lr=0.0002, beta_1=0.5)

    def define_discriminator(self):
        '''Define discriminator structure'''
        discriminator=Sequential()
        discriminator.add(Dense(units=1024, input_dim=784))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dropout(0.3))
        discriminator.add(Dense(units=512))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dropout(0.3))
        discriminator.add(Dense(units=256))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dense(units=1, activation='sigmoid'))
        discriminator.compile(loss='binary_crossentropy', optimizer=self.adam_optimizer())
        return discriminator

    def define_generator(self):
        '''Define generator structure'''
        generator=Sequential()
        generator.add(Dense(units=256, input_dim=100))
        generator.add(LeakyReLU(0.2))
        generator.add(Dense(units=512))
        generator.add(LeakyReLU(0.2))
        generator.add(Dense(units=1024))
        generator.add(LeakyReLU(0.2))
        generator.add(Dense(units=784, activation='tanh'))
        generator.compile(loss='binary_crossentropy', optimizer=self.adam_optimizer())
        return generator

    # define a composite gan model for the generator and discriminator
    def define_gan(self, generator, discriminator):
        '''Define gan structure for training'''
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
    def train_gan(self, dataset):
        '''Define the gan training pipeline'''
        for i in range(self.n_steps):
            # generate points in the latent space
            z = np.random.randn(self.latent_dim * self.n_batch)
            # reshape into a batch of inputs for the network
            z = z.reshape(self.n_batch, self.latent_dim)
            # generate fake images
            X_fake = self.generator.predict(z)
            # select a batch of random real images
            ix = np.random.randint(0, self.size, self.n_batch)
            # retrieve real images
            X_real = dataset[ix]
            # update weights of the discriminator model
            # define target labels for fake images
            y_fake = np.zeros((self.n_batch, 1))
            self.discriminator.trainable = True
            # update the discriminator for fake images
            self.discriminator.train_on_batch(X_fake, y_fake)
            # define target labels for real images
            y_real = np.ones((self.n_batch, 1))
            # update the discriminator for real images
            self.discriminator.train_on_batch(X_real, y_real)

            # generate points in the latent space
            z = np.random.randn(self.latent_dim * self.n_batch)
            # reshape into a batch of inputs for the network
            z = z.reshape(self.n_batch, self.latent_dim)
            # define target labels for real images
            y_real = np.ones((self.n_batch, 1))
            # update generator model
            self.gan.train_on_batch(z, y_real)
