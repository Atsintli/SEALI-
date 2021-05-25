#%%
# -*- coding: utf-8 -*-
import numpy as np
from keras.utils import np_utils
import pandas as pd
import keras as K
from keras.models import *
from keras.layers.core import *
from keras.optimizers import RMSprop, Adam
from keras.layers import LSTM
from keras.layers import BatchNormalization as BatchNorm
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import pickle
np.random.seed(500)
from scipy.stats import expon, randint
from keras.utils import np_utils
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def parse_database():
  trainset = []
  with open("derekbailey_sbic.csv") as archivo:
    #flat = archivo.read().splitlines()
    lineas = archivo.read().splitlines()
    for l in lineas:
      linea = l.split(' ')
      floats = [float(i) for i in linea]
      trainset.append(floats)
  return trainset

x = parse_database()

x = np.array(x)
#print(x.shape)
#print(x[0:5])
x = np.round(x, 2)
x2 = np.round(x, 2)
x2 = x2.flatten().tolist()
#print("soy x2:", x2)
x = np.unique(x) #los diferentes valores dentro de la secuencia de datos equivalente a set 
print("uniques:", len(x))
#print(x)

note_to_int = dict((note, number) for number, note in enumerate(x))

type(note_to_int)
note_to_int
#%%
def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

    # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(x))
    #print(note_to_int)

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        #print((sequence_in))
        sequence_out = notes[i + sequence_length] #51, 52
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])
    #print(network_input)
    #print(network_output)

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    #print(np.shape(network_input))
    #print((network_input))
    # normalize input
    #print(network_input[0:5]) #genera la secuencia de valores basados en la asignación hecha en note_to_int 
    network_input = network_input / float(n_vocab) # normalize values
    print(network_input[0:5])

    network_output = np_utils.to_categorical(network_output)
    print(network_output[0:5])

    return network_input, network_output

n_vocab = len(set(x2)) #x es 
#print(n_vocab)
#print(len(flatness))
#print("secuencia original", x2)
prepare_sequences(x2, n_vocab)

# %%
new_model = load_model("saved_models/weights/weights-improvement-1518-0.6821-bigger.hdf5"
)
network_input, network_output = prepare_sequences(x2, n_vocab)
#new_model.evaluate(network_input,network_output)

# %%
#Continue training from checkpoint

def train(new_model, network_input, network_output):
    filepath = "saved_models/weights/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]
    new_model.fit(network_input, network_output, epochs=5000, batch_size=512, callbacks=callbacks_list)

def train_network():
    n_vocab = len(set(x2))
    network_input, network_output = prepare_sequences(x2, n_vocab)
    train(new_model, network_input, network_output)

if __name__ == '__main__':
    train_network()