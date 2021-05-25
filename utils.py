from numpy import savetxt
import json
import glob
import librosa
import os
import json
import essentia
import essentia.standard as ess
from essentia.standard import *
import matplotlib.pyplot as plt
import numpy as np
from numpy import savetxt
import csv
import toolz as tz

def get_json(filename):
    resultF = open(filename, 'r')
    result = resultF.read()
    #resultF.close()
    return json.loads(result)

def save_matrix_array(file_name, matrixes):
    with open(file_name, 'w') as f:
        for matrix in matrixes:
            savetxt(f, matrix)

def save_as_json(file_name, data):
    with open(file_name, 'w') as f:
        f.write(json.dumps(data))
        f.close()

def save_descriptors_as_matrix(file_name, features):
	with open(file_name, 'w') as f:
	    #m = np.matrix(features)
	    savetxt(f, features, fmt='%s') #fmt='%s'
	print('Save Descriptors as Matrix: Done')

###Write files by the number of clusters###
def writeFiles(n_clusters_, labels):
    folder = 'Clusters/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    files = []

    for audio_files in sorted(glob.glob( 'Segments/' + "*.wav" )):
        names = audio_files.split('/')[1]
        #print(names)
        files.append(names)

    with open('archivos_clases.txt', 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerows(zip(files, labels))

    # concatenate classes in archives
    clases = open('archivos_clases.txt')
    clasescontent = clases.readlines()
    clases = [int(x.split(" ")[1]) for x in clasescontent]

    for clase in range(n_clusters_):
        print("iterando sobre " + str(clase))
        ele = np.where(np.array(clases) == clase)[0]
        print("indices de clase " + str(clase) + " son: " + str(ele))
        #print(ele)
        audiototal = np.array([])
        for elements in ele:
            num = 'Segments/{:06d}'.format(elements)
            for audio_files in sorted(glob.glob(num + "*.wav")):
                print("Escribiendo " + audio_files)
                y, sr = librosa.load(audio_files)
                audiototal = np.append(audiototal, y)
                #print(audiototal)
                librosa.output.write_wav("Clusters/" + "CLASE_"
                                         + str(clase) + ".wav", audiototal, sr)

def extract_mfccs(audio_file):
    loader = essentia.standard.MonoLoader(filename=audio_file)
    print("Analyzing:" + audio_file)
    audio = loader()
    spectrum = Spectrum()
    melBands = MelBands()
    w = Windowing(type='hann')
    fft = FFT()

    name = audio_file.split('/')[1].split('.')[-2]
    print(name)

    pool = essentia.Pool()
    for frame in ess.FrameGenerator(audio, frameSize=2048, hopSize=2048, startFromZero=True): #for chroma frameSize=8192*2, hopSize=8192, #fz=88200, hs=44100
        mag, phase, = CartesianToPolar()(fft(w(frame)))
        mfcc_bands, mfcc_coeffs = MFCC(numberCoefficients=13)(mag)
        loudness = Loudness()(mag)
        contrast, spectralValley = SpectralContrast()(mag)
        flatness = Flatness()(mag) 
        centroid = Centroid()(mag)
        spectral_complex = SpectralComplexity()(mag)
        #mel_bands = melBands(spectrum(w(frame)))
        #dynamic_complexity, loudness = DynamicComplexity()(mag)
        #croma = Chromagram(sampleRate=2048*5)(mag[1:],)
        #onset = OnsetDetection()(mag,phase)
                   
        #['flatness', 'mfccVar','complexity','mfccMean','loudness','centroid','spectralContrast'],

        pool.add('lowlevel.mfcc', mfcc_coeffs)
        pool.add('lowlevel.loudness', [loudness])
        pool.add('lowlevel.spectralcontrast', contrast)
        pool.add('lowlevel.flatness', [flatness])
        pool.add('lowlevel.spectral_complexity', [spectral_complex])
        pool.add('lowlevel.centroid', [centroid])
        #pool.add('lowlevel.onsets', [onset])
        #pool.add('lowlevel.melbands', mel_bands)
        #pool.add('lowlevel.dyncomplex', [dynamic_complexity])
        #pool.add('lowlevel.chroma', croma)
        #pool.add('lowlevel.dens', dens)
    
    pool.add('audio_file', (name))
    aggrPool = PoolAggregator(defaultStats=['mean','var'])(pool)

    YamlOutput(filename='features.json', format='json',
               writeVersion=False)(aggrPool)

    json_data = get_json("features.json")
    #dyncomp = json_data['lowlevel']['dynamic_complexity']['mean']

    #SCMIR Audio Features
    #[[MFCC],[Chromagram],[SpecPcile, 0.95],[SpecPcile, 0.80],[SpecFlatness]];
    #['flatness', 'mfccVar','complexity','mfccMean','loudness','centroid','spectralContrast'],


    #os.remove("mfccmean.json")
    return {"file": json_data['audio_file'],
            "flatness": json_data['lowlevel']['flatness']['mean'],
            "mfccVar": json_data['lowlevel']['mfcc']['var'],
            "complexity": json_data['lowlevel']['spectral_complexity']['mean'],
            "mfccMean": json_data['lowlevel']['mfcc']['mean'],
            "loudness": json_data['lowlevel']['loudness']['mean'],
            "centroid": json_data['lowlevel']['centroid']['mean'],
            "spectralContrast": json_data['lowlevel']['spectralcontrast']['mean'],
            #"mel": json_data['lowlevel']['melbands']['mean'],
            # "chroma": json_data['lowlevel']['chroma']['mean'],
            #"onsets": json_data['lowlevel']['onsets']['mean'],
            #"dyncomplexity": json_data['lowlevel']['dyncomplex']['mean'],
            #"dens": json_data['lowlevel']['dens']['mean'],
            #"densVar": json_data['lowlevel']['dens']['var'],
            }

def extract_all_mfccs(audio_files):
    print("Extracting Features")
    return list(map(extract_mfccs, audio_files))

def getProps(props, dict):
    return map(lambda prop: dict[prop], props)

def concat_features(input_data):
    features = list(map(lambda data: 
               list(tz.concat(getProps(
                   #['flatness', 'complexity', 'dyncomplexity','mfccMean','onsets'],
                   ['flatness', 'mfccVar','complexity','mfccMean','loudness','centroid','spectralContrast'],
                   #['mfccMean','flatness', 'complexity', 'onsets'],
                   #['mfccMean', 'mfccVar'],
                   #['loudness', 'file'],
                   data))),
    input_data))
    #print(features)
    return features

def save_as_matrix(features):
    save_descriptors_as_matrix('database_names.csv', features)

#test

#input_data = extract_all_mfccs(sorted(glob.glob('Segments/' + "*.wav")))
#print(input_data)
#save_as_matrix(concat_features(input_data))
            #print(audiototal)
