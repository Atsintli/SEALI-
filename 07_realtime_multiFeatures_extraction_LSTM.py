#%%
import numpy as np
import matplotlib.pyplot as plt
import soundcard as sc
from struct import unpack
#from IPython import display
from essentia.streaming import *
from essentia import Pool, run, array, reset
from scipy.special import softmax
from essentia import INFO

#OSC libs
import argparse
import math
import requests # importing the requests library 
from pythonosc import dispatcher
from pythonosc import osc_server
from pythonosc import udp_client
import json
from sklearn.preprocessing import MinMaxScaler
from essentia.streaming import FlatnessSFX as sFlatnessSFX

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
from keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout

#%%
min_max_scaler = MinMaxScaler()

sampleRate = 44100
frameSize = 2048 
hopSize = 2048
numberBands = 13
onsets = 1
loudness_bands = 1
# analysis parameters
patchSize = 1  #control the velocity of the extractor 20 is approximately one second of audio

bufferSize = patchSize * hopSize
buffer = np.zeros(bufferSize, dtype='float32')
vectorInput = VectorInput(buffer)
frameCutter = FrameCutter(frameSize=frameSize, hopSize=hopSize)
w = Windowing(type = 'hann')
spec = Spectrum()
mfcc = MFCC(numberCoefficients=numberBands)
fft = FFT() # this gives us a complex FFT
c2p = CartesianToPolar()
onset = OnsetDetection()
eqloud = EqualLoudness()
flatness = Flatness()
envelope = Envelope()
accu = RealAccumulator()
loudness = Loudness()
complexity = SpectralComplexity()
centroid = Centroid()
square = UnaryOperator(type='square')

load_model = "saved_models/weights/movil_FLC_100TS-64N_322_0.0284.hdf5"

def multifeaturesExtractor():
    pool = Pool()

    vectorInput.data  >> frameCutter.signal # al centro puede ir un >> eqloud.signal
    frameCutter.frame >> w.frame >> spec.frame
    spec.spectrum     >> flatness.array
    frameCutter.frame >> loudness.signal 
    spec.spectrum     >> centroid.array #al centro puede ir un >> square.array que cambios genera?
    spec.spectrum     >> mfcc.spectrum
    #spec.spectrum     >> complexity.spectrum

    flatness.flatness >> (pool, 'flatness')
    loudness.loudness >> (pool, 'loudness')
    centroid.centroid >> (pool, 'centroid')
    mfcc.mfcc         >> (pool, 'mfcc')
    mfcc.bands        >> None
    
    #complexity.complexity >> (pool, "spectral complexity")

    #w.frame           >> fft.frame
    #fft.fft           >> c2p.complex
    #c2p.magnitude     >> onset.spectrum
    #c2p.phase         >> onset.phase
    #onset.onsetDetection >> (pool, 'onset')
    
    return pool 

pool = multifeaturesExtractor()

def callback(data):
    # update audio buffer
    buffer[:] = array(unpack('f' * bufferSize, data))
    #print ("this is the buffer", buffer[:])
    flatnessBuffer = np.zeros([1])
    loudnessBuffer = np.zeros([1])
    centroidBuffer = np.zeros([1])
    mfccBuffer = np.zeros([numberBands])
    #onsetBuffer = np.zeros([onsets])

    reset(vectorInput)
    run(vectorInput)

    flatnessBuffer = np.roll(flatnessBuffer, -patchSize)
    loudnessBuffer = np.roll(loudnessBuffer, -patchSize)
    centroidBuffer = np.roll(centroidBuffer, -patchSize)
    mfccBuffer = np.roll(mfccBuffer, -patchSize)
    #onsetBuffer = np.roll(onsetBuffer, -patchSize)


    flatnessBuffer = pool['flatness'][-patchSize]
    loudnessBuffer = pool['loudness'][-patchSize]
    centroidBuffer = pool['centroid'][-patchSize]
    mfccBuffer = pool['mfcc'][-patchSize]
    #onsetBuffer = pool['onset'][-patchSize]

    #print ("MFCCs:", '\n', (mfccBuffer))
    #print ("OnsetDetection:", '\n', onsetBuffer)
    features = np.concatenate((flatnessBuffer, loudnessBuffer, centroidBuffer, mfccBuffer), axis=None)
    features = features.tolist()
    return features

#%%
def create_network():
    """ create the structure of the neural network """
    model = Sequential()
    neurons = 150
    model.add(LSTM(
        neurons,
        input_shape=(100, 16), #n_steps, n_features
        return_sequences=False
    ))
    model.add(RepeatVector(16)) #n_features
    model.add(LSTM(neurons, return_sequences=True))
    model.add(TimeDistributed(Dense(11, activation='softmax')))
    model.compile(loss='categorical_crossentropy' , optimizer='adam', metrics=[ 'accuracy' ])
    model.load_weights(load_model)
    return model

all_features = []
model = create_network()
min_max_scaler = MinMaxScaler()

#%%

def set_zero(sample, d, val):
    """Set all max value along dimension d in matrix sample."""
    argmax_idxs = sample.argmax(d)
    return argmax_idxs*0.1 #this can vary depends decimal transform of np.round

def accFeatures(data): 
    
    global model
    global all_features
    
    i = len(all_features)
    if i == 100: #n_steps requiered by the model
        all_features = np.array(all_features)
        features_scaled = min_max_scaler.fit_transform(all_features)
        #print("features_scaled: \n", features_scaled[0:6])
        features_scaled = np.round(features_scaled, 1)
        features_reshaped = np.reshape(features_scaled, (1, features_scaled.shape[0], features_scaled.shape[1]))
        #print("allFeatures:", all_features)
        prediction = model.predict(features_reshaped, verbose=0) #shape = (n, 1) where n is the n_days_for_prediction
        prediction = set_zero(prediction, d=2, val=0)
        print("Prediction: \n", prediction)
        all_features = []
    else:
        all_features.append(data)

# capture and process the speakers loopback
# the 2 selects the external interface Zoom h5 #3 for jack
with sc.all_microphones(include_loopback=True)[3].recorder(samplerate=sampleRate) as mic:
  while True: accFeatures(callback(mic.record(numframes=bufferSize).mean(axis=1)))

#with sc.all_microphones(include_loopback=True)[3].recorder(samplerate=sampleRate) as mic:
#    for i in range(1000): 
#      accFeatures(callback(mic.record(numframes=bufferSize).mean(axis=1)))
# %%
  