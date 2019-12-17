"""
Minimal Fully Connected Autoencoder

This Autoencoder is a simple single hidden layer fully connected
autoencoder applied to the MNIST dataset.

This version has been adapted from Francois Chollet

This is constrained by the size of the hidden layers' size
hence what typically happens in the hidden layer is that
it learns an approximation of PCA. but this version adds another
sparsity constraint to make the representation compact. The sparsity
constraints acts on the activity of the hidden representations so 
fewer units would fire at a given time. In Keras, this can be done by 
adding an activity regularizer on the embedding layer

Author: Paul Scherer
Date: March 2019
"""

from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers

from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt


# Data
################
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print x_train.shape
print x_test.shape

# Hyperparameter
#################
encoding_dim = 32
input_img = Input(shape=(784,))

# The encoded representation of the input
encoded = Dense(encoding_dim, activation="relu", activity_regularizer=regularizers.l1(10e-5))(input_img)
# The decoded - lossy reconstruction of the input
decoded = Dense(784, activation="relu")(encoded)

# Instantiate the autoencoder we made
autoencoder = Model(input_img, decoded)


## Seperate encoder 
# that maps an input to its encoded representation
encoder = Model(input_img, encoded)

## Seperate Decoder 
# that maps the hidden to the input
# placeholder for an encoded (32-dim) input)
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))


autoencoder.compile(optimizer = "adadelta", loss="binary_crossentropy")
autoencoder.fit(x_train, x_train, epochs=100, batch_size=256, shuffle=True, validation_data=(x_test,x_test))

# encode and decode some digits using the seperate encoder and decoder 
# after their layers had been trained by the autoencoder
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

# For visualization of the images
n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()