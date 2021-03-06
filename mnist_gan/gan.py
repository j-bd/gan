#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 11:19:52 2020

@author: j-bd
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import adam


class Gan:
    '''gan implementation'''
    def __init__(self, dataset_size, latent_dim, n_epochs, n_batch):
        '''Instance a new gan object'''
        self.latent_dim = latent_dim
        self.n_batch = n_batch
        self.size = dataset_size

        self.batches_per_epoch = int(self.size / n_batch)
        self.n_steps = self.batches_per_epoch * n_epochs

        self.generator = self.define_generator()
        self.generator.summary()
        self.discriminator = self.define_discriminator()
        self.discriminator.summary()
        self.gan = self.define_gan(self.generator, self.discriminator)
        self.gan.summary()

        self.loss_fake_d = list()
        self.loss_real_d = list()
        self.loss_gan = list()

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

    def loss_computation(self, list_hist):
        '''Compute and save model loss'''
        plt.figure("Training history", figsize=(15.0, 5.0))
        for element in list_hist:
            position = 131 + list_hist.index(element)
            plt.subplot(position)
            plt.plot(range(1, len(element[0]) + 1), element[0], label=element[1])
            plt.title("Loss function evolution")
            plt.legend()
            plt.xlabel("Number of iterations")
            plt.ylabel("Loss value")
        plt.show()
        plt.savefig(f"gan_mnist-steps{self.n_steps}-training.png")

    # gan training algorithm
    def train_gan(self, dataset):
        '''Define the gan training pipeline'''
        for i in range(self.n_steps):
            # generate points in the latent space
            latent_space = np.random.randn(self.latent_dim * self.n_batch)
            # reshape into a batch of inputs for the network
            latent_space = latent_space.reshape(self.n_batch, self.latent_dim)
            # generate fake images
            x_fake = self.generator.predict(latent_space)
            # select a batch of random real images
            real_sample = np.random.randint(0, self.size, self.n_batch)
            # retrieve real images
            x_real = dataset[real_sample]
            # update weights of the discriminator model
            # define target labels for fake images
            y_fake = np.zeros((self.n_batch, 1))
            self.discriminator.trainable = True
            # update the discriminator for fake images
            hist_d_fake = self.discriminator.train_on_batch(x_fake, y_fake)
            self.loss_fake_d.append(hist_d_fake)
            # define target labels for real images
            y_real = np.ones((self.n_batch, 1))
            # update the discriminator for real images
            hist_d_real = self.discriminator.train_on_batch(x_real, y_real)
            self.loss_real_d.append(hist_d_real)

            # generate points in the latent space
            latent_space = np.random.randn(self.latent_dim * self.n_batch)
            # reshape into a batch of inputs for the network
            latent_space = latent_space.reshape(self.n_batch, self.latent_dim)
            # define target labels for real images
            y_real = np.ones((self.n_batch, 1))
            # update generator model
            hist_g = self.gan.train_on_batch(latent_space, y_real)
            self.loss_gan.append(hist_g)

        #Loss computation
        print(self.loss_fake_d)
        loss_list = [
            (np.mean(self.loss_fake_d), "discriminator_fake_detection"),
            (np.mean(self.loss_real_d), "discriminator_real_detection"),
            (np.mean(self.loss_gan), "gan_detection")
        ]
        self.loss_computation(loss_list)
