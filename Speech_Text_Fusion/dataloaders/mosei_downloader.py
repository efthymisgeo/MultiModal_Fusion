from Speech_Text_Fusion.mmsdk import mmdatasdk
from read_dataset import ReadDlData
import numpy as np
import os

# TODO change cmumosi to cmumosei - already downloaded - to be aligned

def add_labels(cmumosi_higlevel):
    cmumosi_highlevel.add_computational_sequences(mmdatasdk.cmu_mosi.labels,'cmumosi/')
    cmumosi_highlevel.align('Opinion Segment Labels')
    deploy_files={x:x for x in cmumosi_highlevel.computational_sequences.keys()}
    cmumosi_highlevel.deploy('aligned/', deploy_files)
    aligned_cmumosi_highlevel=mmdatasdk.mmdataset('aligned/')
    return(cmumosi_higlevel)

def myavg2(intervals,features):
    return np.average(features,axis=0)

def directory_check(dirName):
    exists = True
    if os.path.exists(dirName) and os.path.isdir(dirName):
        if not os.listdir(dirName):
            print("MOSI Directory is empty")
            exists = False
        else:
            print("MOSI Directory is not empty")
    else:
        print("MOSI Directory doesn't exist")
        exists = False
    return(exists)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cmumosi_path = os.path.join(BASE_PATH, 'cmumosi')
aligned_path = os.path.join(BASE_PATH, 'aligned')
print("preaparing CMU MOSI download")

if not directory_check(cmumosi_path):
    ## execute the following command only once
    print(".....mosi is being downloaded")
    cmumosi_highlevel = mmdatasdk.mmdataset(mmdatasdk.cmu_mosi.highlevel,
                                            'cmumosi/')
else:
    print("mosi is already downloaded")
    cmumosi_highlevel = ReadDlData(mmdatasdk.cmu_mosi.highlevel, cmumosi_path)


cmumosi_highlevel.align('glove_vectors',collapse_functions=[myavg2])
cmumosi_highlevel.add_computational_sequences(mmdatasdk.cmu_mosi.labels,'cmumosi/')
size_list = [9216, 74, 47, 300, 1585]

cmumosi_highlevel.align('Opinion Segment Labels')
deploy_files={x:x for x in cmumosi_highlevel.computational_sequences.keys()}
cmumosi_highlevel.deploy('aligned/', deploy_files)
aligned_cmumosi_highlevel=mmdatasdk.mmdataset('aligned/')


#cmumosi_highlevel = add_labels(cmumosi_highlev)
    




























