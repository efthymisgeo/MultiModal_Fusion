import os
from Speech_Text_Fusion.mmsdk import mmdatasdk
from os import listdir
from os.path import isfile, join


def SearchDictKeys(dict, name_list):
    short_names = {}
    for csd_name in name_list:
        for k in dict:
            if csd_name in k:
                short_names[csd_name] = dict[k]
                continue
    return(short_names)

def Load_Aligned_Data():
    '''Function that read already downloaded files.
    Returns a mmdatasdk class with speech and text
    multimodal data features'''

    dataset_set = {"glove_vectors.csd", "COVAREP.csd", "Opinion Segment Labels.csd"}
    dataset_dictionary={}
    BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(BASE_PATH, 'aligned')

    if os.path.isdir(path) is False:
        print ("Folder does not exist ...")
        exit(-1)

    csdfiles = [f for f in listdir(path) if isfile(join(path, f)) and f[-4:]=='.csd' and (f in dataset_set)]
    if len(csdfiles)==0:
        print("No csd files in the given folder")
        exit(-2)


    print("%d csd files found"%len(csdfiles))
    for csdfile in csdfiles:
        dataset_dictionary[csdfile]=os.path.join(path,csdfile)
    dataset=mmdatasdk.mmdataset(dataset_dictionary)

    print ("List of the computational sequences")
    print (dataset.computational_sequences.keys())


    return(dataset)

def h5_to_numpy(h5_feats):
    '''function that gets an h5py dict as input
    and converts it into a numpy array'''
    np_arr = np.zeros(h5_feats.shape)
    h5_feats.read_direct(np_arr)
    return np_arr

if __name__ == "__main__":
    # execute only if run as a script
    aligned_dataset = Load_Aligned_Data()
    print("Aligned Dataset Succesfully Loaded")

    for key in keys:
        word_embd.append(h5_to_numpy(glove[key]['features']))
        acoustic_fts.append(h5_to_numpy(covarep[key]['features']))
        targets.append(h5_to_numpy(opinions[key]['features']))

    converted_data = (word_embd, acoustic_fts, targets)

    for i,modal in enumerate(modals):
        pickle_save(modal, converted_data[i])

    i = 47
    key = i
    print('------------extracted data structure-------------')
    print('GloVe Embedding Example:')
    print(word_embd[key].shape, type(word_embd[key]))
    print(word_embd[key])
    print('COVAREP EMbeddings:')
    print(acoustic_fts[key].shape, type(acoustic_fts[key]))
    print(acoustic_fts[key])
    print('Opinions')
    print(targets[key].shape, type(targets[key]))
    print(targets[key])

    s_neg = 0
    neg = 0
    w_neg = 0
    neut = 0
    w_pos = 0
    pos = 0
    s_pos = 0
    counter = 0

    for i, _ in enumerate(keys):
        counter +=1
        s = round(targets[i])
        if s ==-3:
            s_neg += 1
        elif s==-2:
            neg += 1
        elif s==-1:
            w_neg += 1
        elif s==0:
            neut += 1
        elif s==1:
            w_pos += 1
        elif s==2:
            pos += 1
        else:
            s_pos += 1

    print(s_neg,neg,w_neg,neut,w_pos,pos,s_pos)

