import torch
from torch.utils.data.dataset import Dataset

import numpy as np

from config import MAX_LEN, DEVICE


class MultiModalDataset(Dataset):
    def __init__(self, mmodal_dict, task='Binary', approach='sequential'):
        '''
        we take a dictionary input
        mmodal_dict:
            'Audio Features' - covarep list of (*,5033) np.arrays
            'Word Embeddings' - glove list of  (*,5033) np.arrays
            'Opinion Labels' - opinion list of (1,1) np.arrays
        OUTPUT:
            (covarep, glove, opinions)
            --- covarep = (acoustic_fts, real_size)
            --- glove
            --- opinions
        '''

        # each class attribute contains a list o features/labels
        _covarep = mmodal_dict["Audio Features"]
        _glove = mmodal_dict["Word Embeddings"]
        _opinions = mmodal_dict["Opinion Labels"]

        self.audio_features, self.real_lengths = self.__get_acoustic_features(_covarep)
        self.word_embeddings, self.real_lens = self.__get_embeddings(_glove)
        self.opinions = self.__get_opinions(_opinions,task)

    def __getitem__(self, index):
        return ((self.audio_features[index], self.real_lengths[index]),
                (self.word_embeddings[index], self.real_lens[index]),
                self.opinions[index])

    def __len__(self):
        return (len(self.opinions))

    def __get_embeddings(self, glove):
        '''
        function that returns tensor word embeddings
        '''
        real_size = []
        word_embeddings = []

        for item, embedding in enumerate(glove):
            # get real size
            size = embedding.shape[0]

            # get real GloVe embedding
            embedding = embedding[:, :300]
            embedding, real_length = self.__zero_pad(embedding)

            # tensors
            word_embeddings.append(torch.from_numpy(embedding).to(DEVICE))
            real_size.append(torch.tensor(real_length).to(DEVICE))

        return (word_embeddings, real_size)

    def __opinion_per_task(self, labels, task):
        # categorize opinions per task
        # either binary clf or 5class clf
        if task == 'Binary':
            for i, opinion in enumerate(labels):
                # y = 0.5(x+1) mapping
                labels[i] = 0.5 * (1 + np.sign(opinion)).astype(int)
            else:
                # fix me to 5 class clf
                pass
        return (labels)

    def __get_opinions(self, labels, task):
        '''
        function that gets us opinion label segments
        '''
        labels = self.__opinion_per_task(labels, task)

        for i, opinion in enumerate(labels):
            labels[i] = torch.from_numpy(opinion).to(DEVICE)
        return (labels)

    def __get_acoustic_features(self, covarep):
        '''
        function that returns tensor (MAX_LEN,74) acoustic feature
        and real length lists
        Args
            covarep: [ ...(size, 5033)...]
        Outputs
            acoustic_features: [...(MAX_LEN,74)...]
            real_size:
        '''
        acoustic_features = []
        real_size = []

        for i, cov_matrix in enumerate(covarep):
            # loop over acoustic features at sentence level
            audio_sequence = []
            for seqs in range(cov_matrix.shape[0]):
                # reshape word level audio features
                word_level_audio = cov_matrix[seqs, :5032].reshape(-1, 74)

                # get real audio word length
                real_len = cov_matrix[seqs, -1]

                # average acoustic features at word level
                audio_feature = np.sum(word_level_audio, axis=0) / real_len
                audio_sequence.append(audio_feature.reshape(1,-1))

            # concatenate sentense acoustic features
            matrix = audio_sequence[0]
            for k in range(1, len(audio_sequence)):
                matrix = np.concatenate((matrix, audio_sequence[k]), axis=0)

            # zero pad to max length
            matrix, real_len = self.__zero_pad(matrix)

            acoustic_features.append(torch.from_numpy(matrix).to(DEVICE))
            real_size.append(torch.tensor(real_len).to(DEVICE))

        return (acoustic_features, real_size)

    def __zero_pad(self, matrix):
        '''
        zero pads a given matrix to (MAX_LEN,*)
        '''
        size = matrix.shape[0]
        dim = matrix.shape[1]

        if size >= MAX_LEN:
            matrix = matrix[:MAX_LEN, :]
            real_size = MAX_LEN
        else:
            zeros = np.zeros((MAX_LEN - size, dim))
            matrix = np.concatenate((matrix, zeros), axis=0)
            real_size = size

        return matrix, real_size

