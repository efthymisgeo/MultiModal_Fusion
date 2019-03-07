import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from utils.Zero_Padder import ZeroPadding

from config import DEVICE



class MModalDataset(Dataset):
    def __init__(self, mmodal_dict, task='Binary',
                 approach='sequential'):
        ''' we take a dictionary input
        mmodal_dict:
            'Audio Features' - covarep list of list of (1,5035) np.arrays
            'Word Embeddings' - glove list of  (*,300) np.arrays
            'Opinion Labels' - opinion list of (1,1) np.arrays
        OUTPUT:
            (covarep, glove, opinions)
            --- covarep = (acoustic_fts, real_size)
            --- glove
            --- opinions                  '''

        # each class attribute contains a list o features/labels
        covarep_ = mmodal_dict["Audio Features"]
        glove_ = mmodal_dict["Word Embeddings"]
        opinion_ = mmodal_dict["Opinion Labels"]

        ##################################
        ###     class attributes       ###
        ### COVAREP, GloVe, Opinions   ###
        ##################################
        _covarep = self.__acoustic_features(covarep_)
        if approach == 'sequential':
            _covarep = self.__averaging(_covarep[0], _covarep[1])
            self.covarep = ZeroPadding(_covarep, 68).zero_padded_seq()
        else:
            # approach is convolutional
            # fix me
            pass

        # glove embeddings
        _glove = self.__word_embds(glove_)
        self.glove = ZeroPadding(_glove, 50).zero_padded_seq()

        #opinions
        self.opinions = self.__opinion_labels(opinion_, task)

    def __getitem__(self, index):
        return((self.covarep[0][index], self.covarep[1][index]),
               (self.glove[0][index], self.glove[1][index]),
               self.opinions[index])

    def __len__(self):
        return len(self.opinions)

    def __word_embds(self, glove):
        ''' function that returns tensor word embeddings'''
        for key, embedding in enumerate(glove):
            embedding = embedding[:,:300]
            glove[key] = torch.from_numpy(embedding).to(DEVICE)
        return(glove)

    def __opinion_per_task(self, labels, task):
        # categorize opinions per task
        # either binary clf or 5class clf
        if task == 'Binary':
            for i, opinion in enumerate(labels):
                # y = 0.5(x+1) mapping
                labels[i] = 0.5*(1+np.sign(opinion)).astype(int)
            else:
                #fix me to 5 class clf
                pass
        return(labels)

    def __opinion_labels(self, labels, task):
        ''' function that gets us opinion label segments'''
        labels = self.__opinion_per_task(labels, task)

        for i,opinion in enumerate(labels):
            labels[i] = torch.from_numpy(opinion).to(DEVICE)
        return(labels)

    def __acoustic_features(self, acoustic_fts):
        ''' function that returns tensor (m,72) acoustic feature
        and real length'''
        sentense_len = []
        for sent_id, features in enumerate(acoustic_fts):
            word_len = []
            for word_id, utt_feat in enumerate(features):
                # get real length
                word_len.append(torch.from_numpy(utt_feat[:,-1]//74).type(torch.int).to(DEVICE))
                # iconic size = actual size + 1
                iconic_size = utt_feat.size
                # reshape acoustic feature sequence
                utt_feat = utt_feat[:,:iconic_size-1].reshape(-1,74)
                # cast to tensor
                acoustic_fts[sent_id][word_id] = torch.from_numpy(utt_feat).to(DEVICE)
            sentense_len.append(word_len)
        return(acoustic_fts, sentense_len)

    def __mean(self, _list, lengths):
        for idx, list_tensors in enumerate(_list):
            real_len = lengths[idx]
            # get real length tensor only
            list_tensors = list_tensors[:real_len, :]
            # geat real mean
            mean_tensor = torch.sum(list_tensors, dim=0) / real_len.double().reshape(1, -1)
            if idx != 0:
                representation = torch.cat((representation, mean_tensor), dim=0)
            else:
                representation = mean_tensor
        return (representation)

    def __averaging(self, features, lengths):
        '''
        function that is used in sequential approach
        in order to get a vector representation for
        each word in the sentense
        Args:
            features - a list of lists of (*,74) tensors
            lengths -  a list of lists of (1,1) tensors
        Output:
            avg - a list of (*,74) tensors
            lens - a list of (1,1) tensors
        '''
        avg = []
        lens = []
        for sent_id, sentense_tensors in enumerate(features):
            # get a list of each word's acoustic length
            word_lens = lengths[sent_id]
            # get mean word tensor representation
            avg.append(self.__mean(sentense_tensors, word_lens))
            # get real sentense length
            lens.append(len(word_lens))
        #return(avg, lens)
        return(avg)













