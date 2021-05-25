import sys
import os
from essentia.standard import *
import numpy as np
import glob
from pylab import plot, show, figure, imshow
import matplotlib.pyplot as plt

in_dir = 'audiotest/'
out_dir = 'segments_audiotest/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

counter = 0

def segments_gen(fileName):
    loader = essentia.standard.MonoLoader(filename=fileName)
    audio = loader()
    print('\n')
    print("Generating Segments: " + fileName)
    print("Num of samples: ", len(audio))

    w = Windowing(type = 'hann')
    spectrum = Spectrum()
    mfcc = MFCC()

    logNorm = UnaryOperator(type='log')
    pool = essentia.Pool()
    for frame in FrameGenerator(audio, frameSize = 8192,  hopSize = 512, startFromZero=True): #2048, 512
        mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
        pool.add('lowlevel.mfcc', mfcc_coeffs)

    #tiny segments 
    minimumSegmentsLength = 1
    size1 = 10
    inc1 = 10
    size2 = 10
    inc2 = 10
    cpw = 1

    features = [val for val in pool['lowlevel.mfcc'].transpose()]
    sbic = SBic(size1=size1, inc1=inc1,size2=size2, inc2=inc2,cpw=cpw, minLength=minimumSegmentsLength)
    segments = sbic(np.array(features))
    record_segments(audio,segments)

def record_segments(audio, segments):
    for segment_index in range(len(segments) - 1):
        global counter
        start_position = int(segments[segment_index] * 512)
        end_position = int(segments[segment_index + 1] * 512)
        writer = essentia.standard.MonoWriter(filename=out_dir + "{:06d}".format(counter) + ".wav", format="wav")(audio[start_position:end_position])
        counter = counter + 1
    print('Num of Segments: ' + str(len(segments)))

def gen_all_segments(audio_files):
	return list(map(segments_gen, audio_files))

input_data = gen_all_segments(sorted(glob.glob(in_dir + "*.wav"))) #can be .mp3

print("Done cutting segments")
