#%%
# -*- coding: utf-8 -*-
import numpy as np
from keras.models import *
from keras.layers.core import *
from keras.layers import LSTM
from keras.models import load_model
from keras.layers import BatchNormalization as BatchNorm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import soundfile
import librosa
import glob
import json
import os

file_in = 'segments_flatness.csv'

def parse_database():
  trainset = []
  with open(file_in) as archivo:
    lineas = archivo.read().splitlines()
    for l in lineas:
      linea = l.split(' ')
      floats = [float(i) for i in linea]
      trainset.append(floats)
  return trainset

def data():
    round = 1
    x = parse_database()
    x = np.array(x)
    x = np.round(x, round)
    all_data = np.round(x, round)
    all_data = all_data.flatten().tolist()
    return all_data

#%%
def prepare_sequences(all_data, uniques):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 10

    # create a dictionary to map pitches to integers
    feature_to_int = dict((note, number) for number, note in enumerate(np.unique(all_data)))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(all_data) - sequence_length, 1):
        sequence_in = all_data[i:i + sequence_length]
        sequence_out = all_data[i + sequence_length] #51, 52
        #network_input.append([feature_to_int[char] for char in sequence_in])
        network_input.append(sequence_in)
        network_output.append(feature_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    #network_input = network_input / float(uniques) # normalize values
    #print(network_input[0:10])

    network_output = np_utils.to_categorical(network_output)
    #print(network_output[0:5])

    return network_input, network_output
#%%
def create_network(timesteps, x, y):
    """ create the structure of the neural network """
    neurons = 64
    model = Sequential()
    model.add(LSTM(
        neurons,
        input_shape=(timesteps, x),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(LSTM(neurons, return_sequences=True, recurrent_dropout=0.3,))
    model.add(LSTM(neurons))
    model.add(BatchNorm())
    model.add(Dropout(0.6))
    model.add(Dense(neurons))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.6))
    model.add(Dense(y))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

def train(model, network_input, network_output):
    """ train the neural network """
    filepath = "saved_models/weights/test-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=5000, batch_size=10, callbacks=callbacks_list)

#%%

def train_network():
    all_data = data()
    n_classes = len(set(all_data))
    network_input, network_output = prepare_sequences(all_data, n_classes)
    note_to_int = dict((note, number) for number, note in enumerate(all_data))
    timesteps = network_input.shape[1]
    x = network_input.shape[2]
    y = network_output.shape[1]
    model = create_network(timesteps, x, n_classes)
    train(model, network_input, network_output)

if __name__ == '__main__':
    train_network()

# %%
