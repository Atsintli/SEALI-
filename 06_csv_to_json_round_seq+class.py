#%%
import json
import numpy as np
import glob
import csv
from numpy import loadtxt
import os
from utils import *
import toolz as tz
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler, MinMaxScaler

file_in = "segments_flatness.csv"
json_file = "segments_flatness_rounded_seq.json"
json_file_2 = "segments_flatness_rounded_class.json"

# Function to convert a CSV to JSON
def convert_write_json(data, json_file):
    with open(json_file, "w") as f:
        f.write(json.dumps(data, sort_keys=True, indent=1, separators=(',', ': '))) 

#%%
def roundFeatures(features):
    myarr=[]
    roundNum = 1
    features_round = np.round(features, roundNum)
    features_unique = np.unique(features_round)
    note_to_int = dict((note, number) for number, note in enumerate(features_unique))
    features = np.round(features, roundNum)
    for i in range(len(features)):
        filename = "{:06d}".format(i)
        features_ = features[i]
        myarr.append({'clase': note_to_int[features_], 'file': filename, 'features': features_})
    return myarr

def round_by_seq():
    features = loadtxt(file_in)
    data = roundFeatures(features.tolist())
    convert_write_json(data, json_file)

def round_by_class():
    features = loadtxt(file_in)
    data = roundFeatures(features.tolist())
    grouped_data = tz.groupby(lambda clases: clases['clase'], data)
    convert_write_json(grouped_data, json_file_2)

round_by_seq()
round_by_class()
print('Done')
# %%
