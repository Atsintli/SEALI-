#%%
import numpy as np
from keras.models import *
from keras.layers.core import *
from keras.layers import LSTM
from keras.models import load_model
from keras.layers import BatchNormalization as BatchNorm
import soundfile
import librosa
import glob
import json
import os
#%%
#Init data
json_input = "derekbailey_sbic_round_3.json" #ordered by sequence
json_input_by_class = "derekbailey_sbic_rounded_3_class.json" #ordered by classes
#load_model = "saved_models/weights/weights-improvement-1723-0.0100-bigger.hdf5" #model with 2 round 75 clases
load_model = "saved_models/weights/weights-improvement-2710-0.0046-bigger.hdf5" # model with 3 round 5nn clases
audio_file_out = "Bailey_2"

with open(json_input) as file:
    jsonData = json.load(file)

#All data ordered by classes
def get_classified_data():
    with open(json_input_by_class) as f:
      jsonData = json.load(f)
    return jsonData

#Retrieve uniques for all data
def uniques():
    all_data = jsonData
    all_features = [f['features'] for f in all_data]
    features_unique = np.unique(all_features)
    return features_unique

#Define initial sequence
def initial_seq():
    data = jsonData[0:100]
    features = [f['features'] for f in data]
    features = np.array([features])
    return features

#%%
def create_network(timesteps,x,y): #timesteps, num dimenssions (features), total class number
    neurons = 256
    """ create the structure of the neural network """
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(timesteps, x),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(LSTM(neurons, return_sequences=True, recurrent_dropout=0.3,))
    model.add(LSTM(neurons))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(neurons))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(len(y)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    model.load_weights(load_model)
    return model

def concatenateFiles(fileName):
    folder = 'Autocomposer/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    files = []

    for i in range(1):
        audiototal = np.array([])
        for elements in fileName:
            num = ('segments_DerekBailey/' + elements)
            for audio_files in sorted(glob.glob(num + "*.wav")):
                print("Escribiendo " + audio_files)
                y, sr = librosa.load(audio_files)
                audiototal = np.append(audiototal, y)
                soundfile.write(folder + "Bailey" + ".wav", audiototal, sr)

def getFile(predicted_class, classified_data):
  files_of_class = classified_data[str(predicted_class)]
  index = np.random.randint(len(files_of_class))
  return files_of_class[index]

def predict_sequence_from_data(features, classes, iterations, classified_data):
    timesteps = features.shape[1]
    x = features.shape[0]
    i= 0
    class_to_feature = dict(enumerate(classes))
    model = create_network(timesteps, x, classes) #timesteps, num dimenssions
    files = []
    prediction_input = np.reshape(features, (1, timesteps, x))
    while len(files) < iterations:
      prediction = model.predict(prediction_input, verbose=0)
      class_ = np.argmax(prediction)
      next_file = getFile(class_, classified_data) #to select a random file form predicted class
      files.append(next_file)
      #print("Class to note:", class_to_feature[class_])
      prediction_input_ = np.append(prediction_input.flatten()[1:], class_to_feature[class_]) #adds data(prediction) to prediction
      #print("Prediction_input:", prediction_input_)
      prediction_input = np.reshape(prediction_input_, (1, timesteps, x))
      #print("Prediction_input reshape:", prediction_input_)7
    return list(map(lambda x: x['features'], files))

#%%
def run_all():
    classes = uniques()
    classified_data = get_classified_data() 
    features = initial_seq()
    lastResult = predict_sequence_from_data(features, classes, 20, classified_data)
    #concatenateFiles(lastResult)
    return print("Last result:", lastResult)

run_all()
# %%
