import torch

from config import DEVICE

class ZeroPadding(object):
    '''
    This is a zero padding class which will be used
    in all different modalities during dataloading
    INPUTS:
    - tensor_in (list) a list of unpadded tensors
    - max_len (int)
    OUTPUTS:
    - padded_tensors (list):
        a list of padded to max_len tensors
    - real_lens (list):
        a list of int tensors containig real sizes
    '''
    def __init__(self, tensor_in, max_len):
        self.tensor_in = tensor_in
        self.max_len = max_len
        # a tuple of zero padded input and real lengths
        self.tensor_out = self.__trim_to_max()

    def zero_padded_seq(self):
        return(self.tensor_out)

    def __get_max(self):
        max = 0
        for feature in self.tensor_in:
            if feature.shape[0] > max:
                max = feature.shape[0]
        return(max)

    def __feature_dim(self, idx=0):
        return(self.tensor_in[idx].shape)

    def __trim_to_max(self):
        '''
        function that returns zero-paded
        sequence along w real lens
        '''
        max = self.__get_max()
        if max < self.max_len:
            max = self.max_len

        tensor_out = []
        real_len = []
        _, features_dim = self.__feature_dim()
        for i,tensor in enumerate(self.tensor_in):
            real_size = tensor.shape[0]
            if real_size > max:
                #trim
                tensor_out.append(tensor[:max])
                real_len.append(max)
            else:
                #zero_pad
                zeros = torch.zeros((max-real_size, features_dim),
                                    dtype=torch.float64).to(DEVICE)
                tensor_out.append(torch.cat((tensor, zeros), 0).to(DEVICE))
                real_len.append(real_size)
        return(tensor_out, real_len)