#%%
import json
import essentia
import essentia.standard as ess
from essentia.standard import *
import matplotlib.pyplot as plt
import numpy as np
from numpy import savetxt
import glob
import csv
import os
from utils import *
import toolz as tz

out_file = 'datasets/segments_flatness.csv'
in_dir = 'segments_audiotest/' + "*.wav"

def extract_mfccs(audio_file):
    loader = essentia.standard.MonoLoader(filename=audio_file)
    name = audio_file.split('/')[1].split('.')[-2]
    print("Analyzing:" + audio_file)
    audio = loader()
    w = Windowing(type='hann')
    fft = FFT()

    pool = essentia.Pool()
    for frame in ess.FrameGenerator(audio, frameSize=2048, hopSize=2048, startFromZero=True): #for chroma frameSize=8192*2, hopSize=8192, #fz=88200, hs=44100
        mag, phase, = CartesianToPolar()(fft(w(frame)))
        flatness = Flatness()(mag)
        #mfcc_bands, mfcc_coeffs = MFCC(numberCoefficients=13)(mag)
        #contrast, spectralValley = SpectralContrast()(mag)
        #spectral_complex = SpectralComplexity()(mag)
        #centroid = Centroid()(mag)
        #loudness = Loudness()(mag)        

        pool.add('lowlevel.flatness', [flatness])
        #pool.add('lowlevel.mfcc', mfcc_coeffs)
        #pool.add('lowlevel.loudness', [loudness])
        #pool.add('lowlevel.spectralContrast', contrast)
        #pool.add('lowlevel.centroid', [centroid])
        #pool.add('lowlevel.spectral_complexity', [spectral_complex])
        #pool.add('lowlevel.dyncomplex', [dynamic_complexity])
        #pool.add('lowlevel.melbands', mel_bands)
        #pool.add('lowlevel.onsets', [onset])
        #pool.add('lowlevel.chroma', croma)
        #pool.add('lowlevel.dens', dens)

    pool.add('audio_file', (name))
    aggrPool = PoolAggregator(defaultStats=['mean','var'])(pool)

    YamlOutput(filename='features.json', format='json',
               writeVersion=False)(aggrPool)

    json_data = get_json("features.json")

    #os.remove("mfccmean.json")
    return {"file": json_data['audio_file'],
            "flatness": json_data['lowlevel']['flatness']['mean'],
            #"mfccMean": json_data['lowlevel']['mfcc']['mean'],
            #"mfccVar": json_data['lowlevel']['mfcc']['var'],
            #"complexity": json_data['lowlevel']['spectral_complexity']['mean'],
            #"loudness": json_data['lowlevel']['loudness']['mean'],
            #"centroid": json_data['lowlevel']['centroid']['mean'],
            #"spectralContrast": json_data['lowlevel']['spectralContrast']['mean']
            #"mel": json_data['lowlevel']['melbands']['mean'],
            # "chroma": json_data['lowlevel']['chroma']['mean'],
            #"onsets": json_data['lowlevel']['onsets']['mean'],
            #"dyncomplexity": json_data['lowlevel']['dyncomplex']['mean'],
            #"dens": json_data['lowlevel']['dens']['mean'],
            #"densVar": json_data['lowlevel']['dens']['var'],
            }

def extract_all_mfccs(audio_files):
    return list(map(extract_mfccs, audio_files))

def getProps(props, dict):
    return map(lambda prop: dict[prop], props)

def concat_features(input_data):
    features = list(map(lambda data:
               list(tz.concat(getProps(
                   #['loudness'],
                   #['mfccMean'],
                   #['mfccMean', 'mfccVar'],
                   #['mfccMean','flatness', 'complexity', 'onsets'],
                   #['flatness', 'mfccVar','complexity','mfccMean','loudness','centroid','spectralContrast'],
                   ['flatness'],
                   data))),
    input_data))
    return features

def save_as_matrix(features):
    save_descriptors_as_matrix(out_file, features)

#Run
input_data = extract_all_mfccs(sorted(glob.glob(in_dir)))
save_as_matrix(concat_features(input_data))
print("Done extracting features")