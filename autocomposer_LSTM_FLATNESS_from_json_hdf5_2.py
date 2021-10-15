#%%
import numpy as np
import matplotlib.pyplot as plt
from essentia.streaming import *
from scipy.special import softmax
from essentia import INFO
#from feature_extract_ import extract_all_mfccs
from functools import reduce
from toolz import assoc
import toolz as tz
import json
import soundfile
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import librosa
import keras as K
from keras.models import *
from keras.layers.core import *
from keras.optimizers import RMSprop, Adam
from keras.layers import LSTM
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from keras.layers import BatchNormalization as BatchNorm
import glob

#json_input = "datasets/flatness_seq_minmaxScaler.json"
#json_input = 'datasets/sinkintoreturn_flatness.json'
#json_input = 'datasets/Movil2.json'#sirve
#load_model = "Models/Movil2-5TS-128Neurons-20482-0.0004-.hdf5"#sirve
#load_model = "saved_models/weights/SIN-5TS-128Neurons-979-0.0112-.hdf5"
#load_model = "Models/flatness-100TS-6193-0.0055-GDB_1024BS.hdf5"
#json_input = "derekbailey_sbic_round_2.json"
#load_model = "Models/weights-improvement-1723-0.0100-bigger.hdf5"
#load_model = "saved_models/weights/test-improvement-01-3.9306-bigger.hdf5"

json_input = "datasets/segments_flatness_rounded_seq.json" #ordered by sequence
load_model = "saved_models/weights/SIR-improvement-97416-0.0215-bigger.hdf5"

def getProps(props, dict):
    return map(lambda prop: dict[prop], props)

def concat_features(input_data):
    features = list(map(lambda data:
               list(tz.concat(getProps(
                ['flatness'],
                   data))),
    input_data))
    np_flatness = np.array(features)
    #print('soy shape features',np_flatness.shape)
    return np_flatness

def create_network(x,y): #x sequence timesteps, y = number of features, number of dimensions
    neurons = 512
    """ create the structure of the neural network """
    model = Sequential()
    model.add(LSTM(
        neurons,
        input_shape=(x, y),
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
    model.add(Dense(46)) # the number of diferent classes in the arrays
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    # Load the weights to each node
    model.load_weights(load_model)
    return model

def reducer( flatness, acc, fileData):
  fileData_ =  np.array(fileData['features'])
  flatness_ = np.array(flatness)
  #print("soy fileData:", fileData_)
  #print("soy flatness", flatness)
  #print("soy len filedata", len(file))
  diff = np.linalg.norm(fileData_ - flatness_) #total diff of array
  #print("soy dif", diff)
  if acc==None:
    #print(tz.assoc(fileData, 'diff', diff))
    return tz.assoc(fileData, 'diff', diff)
  else:
    if acc['diff'] <= diff:
        #print('soy acccccccc:', acc)
        return acc
    else:
        return tz.assoc(fileData, 'diff', diff)

def getClosestCandidate(flatness):
  with open(json_input) as f:
      jsonData = json.load(f)
  return reduce(lambda acc, fileData: reducer(flatness, acc, fileData), jsonData, None)

def dedupe(tracklist):
    acc = []
    for el in tracklist:
        if len(acc) == 0:
            acc.append(el)
            continue;
        if acc[-1]["file"] != el["file"]:
            acc.append(el)
    return acc

def concatenateFiles(fileName):
    folder = 'Autocomposer/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    files = []

    for i in range(1):
        audiototal = np.array([])
        for elements in fileName:
            num = ('SIR_audiotestsplits/' + elements)
            for audio_files in sorted(glob.glob(num + "*.wav")):
                print("Escribiendo " + audio_files)
                y, sr = librosa.load(audio_files)
                audiototal = np.append(audiototal, y)
                soundfile.write(folder + "Bailey_test" + ".wav", audiototal, sr)

#%%
#Inicializar datos
with open(json_input) as file:
    jsonData = json.load(file)

all_data = jsonData
all_features = [f['clase'] for f in all_data] #antes feature
features_unique = np.unique(all_features)

data = jsonData[0:5] # from 0 to 99
features = [f['clase'] for f in data] #antes feature
features = np.array([features])

print(features)
print((features.shape))

model = create_network(features.shape[1], features.shape[0]) #timesteps, num dimenssions
#model = create_network(100, 1) #timesteps, num dimenssions (features)

#%%

#class_to_feature = dict(enumerate(features_unique))
#class_to_feature = dict((number, note) for number, note in enumerate(features_unique))
#print(class_to_feature)
#prediction_input = class_to_feature[features] #adds data(prediction) to prediction
#print(prediction_input)

prediction_input = np.reshape(features, (1, features.shape[1], features.shape[0]))#ojo por algun motivo estos datos quedaron al revés respecto a la serie que extraigo directamente de los archivos de audio

#indata = [[[38],[37],[36],[38],[29]]]
#prediction = model.predict(indata, verbose=0)
prediction = model.predict(prediction_input, verbose=0)
##class_to_note[class_]
print(prediction_input)
class_prediction = np.argmax(prediction)
print("class:", class_prediction)

#Le tengo que dar features convertidos a clases!!! 
#despues la clase se convierte a feature y los 5 features vuelven a entrar como clase!

#%%
class_to_note = dict(enumerate(features_unique))
i=0
prediction_input = np.reshape(features, (1, features.shape[1], features.shape[0]))#ojo por algun motivo estos datos quedaron al revés respecto a la serie que extraigo directamente de los archivos de audio

prediction_acc = []
prediction_class = []

while i < 10:
    prediction = model.predict(prediction_input, verbose=0)
    class_ = np.argmax(prediction)
    print("Class:", [class_])
    print("Class to note:", class_to_note[class_])
    prediction_input_ = np.append(prediction_input.flatten()[1:], class_) #adds data(prediction) to prediction
    #print("soy prediction_input:", prediction_input)
    prediction_input = np.reshape(prediction_input_, (1, features.shape[1], features.shape[0]))
    #print("soy prediction_input reshape:", prediction_input_)
    prediction_class.append(class_)
    #prediction_acc.append([class_])
    i = i+1

#%%
print(prediction_class)
result = list(map(getClosestCandidate, prediction_class))#[0:1174]#se esta leyendo n veces el archivo?
#print(prediction_acc)
#result = list(map(getClosestCandidate, prediction_input_))[0:1174]#se esta leyendo n veces el archivo?
#print(len(prediction_acc))
#result = list(map(getClosestCandidate, prediction_acc))#[0:60]#se esta leyendo n veces el archivo?
print (list(map(lambda x: x['file'], result)))
#len(result)
#print(len(result))
#print("result", result)

#lastResult = dedupe(result)
#lastResult = list(map(lambda x: x['file'], lastResult))
#print("last list", lastResult)
#generate_music()
#concatenateFiles(lastResult)

# %%
data = jsonData[10:895]
features = [f['clase'] for f in data]
#print (features)

import matplotlib.pyplot as plt
plt.plot(features)
plt.plot(prediction_class)
plt.ylabel('Clases')
plt.xlabel('Iteraciones')
plt.savefig('100_prediccion_0-100.png')
plt.show()

# %%
