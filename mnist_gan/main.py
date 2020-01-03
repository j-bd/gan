#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 16:00:04 2019

@author: j-bd
"""

from keras.datasets import mnist
import numpy as np


def load_data():
    '''Load mnist data'''
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.astype(np.float32) - 127.5)/127.5

    # convert shape of x_train from (60000, 28, 28) to (60000, 784)
    # 784 columns per row
    x_train = x_train.reshape(60000, 784)
    return (x_train, y_train, x_test, y_test)

def main():
    '''Select dataset'''
    (X_train, y_train,X_test, y_test) = load_data()
    print(X_train.shape)

if __name__ == "__main__":
    main()
