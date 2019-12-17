"""
seq2seq AutoEncoder

If the input is a sequence and you want to output a sequence, 
then you may want to use an encoder decoder type of model to capture
a temproal data manifold in the form of a RNN, specifically an LSTM.
To build a LSTM based autoencoder, first use a LSTM encoder to turn
the input sequence into a single vector that contains information 
about the entire sequence, then repeat this vector n times (where 
n is the number of time steps in the output sequence) then run a LSTM
decoder to turn this constant sequence in to the target sequence

This is code for furture reference
"""

from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model

inputs = Input(shape=(timesteps, input_dim))
encode = LSTM(latent_dim)(inputs)

decoded = RepeatVector(timesteps)(encoded)
decoded = LSTM(input_dim, return_sequences=True)(decoded)

sequence_autoencoder = Model(inputs, decoded)
encoder = Model(inputs, encoded)