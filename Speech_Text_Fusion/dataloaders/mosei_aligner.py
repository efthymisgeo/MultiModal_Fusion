from Speech_Text_Fusion.mmsdk import mmdatasdk
from read_dataset import ReadDlData
import numpy as np
import os


def average(intervals, features):
    return np.average(features, axis=0)


def directory_check(dirName):
    exists = True
    if os.path.exists(dirName) and os.path.isdir(dirName):
        if not os.listdir(dirName):
            print("MOSEI Directory is empty")
            exists = False
        else:
            print("MOSEI Directory is not empty")
    else:
        print("MOSEI Directory doesn't exist")
        exists = False
    return exists


cmumosei_path = '/ssd/speech_data/cmumosei/'

if directory_check(cmumosei_path):
    # aling CMU-MOSEI
    cmumosei_highlevel = ReadDlData(mmdatasdk.cmu_mosei.highlevel, cmumosei_path)

    cmumosei_highlevel.align('glove_vectors', collapse_functions=[average])
    cmumosei_highlevel.add_computational_sequences(mmdatasdk.cmu_mosei.labels, 'cmumosei/')

    cmumosei_highlevel.align('Sentiment Labels')    # 'Emotion Labels' available as well
    deploy_files = {x:x for x in cmumosei_highlevel.computational_sequences.keys()}
    cmumosei_highlevel.deploy('mosei_aligned/', deploy_files)
    aligned_cmumosei_highlevel = mmdatasdk.mmdataset('mosei_aligned/')
