##############################
### Tensor Fusion Network  ###
##############################

# code provided by:
# https://github.com/Justin1904/TensorFusionNetworks/blob/master/model.py
# modified for our needs

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class TFN(nn.Module):
    '''
    Implements the Tensor Fusion Networks for multimodal sentiment analysis as is described in:
    Zadeh, Amir, et al. "Tensor fusion network for multimodal sentiment analysis." EMNLP 2017 Oral.
    '''

    def __init__(self, hidden_dims, dropouts, post_fusion_dim, task='binary'):
        '''
        Args:
            hidden_dims - a length-3 tuple :(audio_dim, video_dim, text_dim)
            dropouts - a length-4 tuple, contains
                        (audio_dropout, video_dropout, text_dropout, post_fusion_dropout)
            post_fusion_dim - (int) size of the sub-networks after tensorfusion
        Output: (return value in forward)
            task:
                - binary: (B,1)
                - 5 class: (B,5)
        '''
        super(TFN, self).__init__()

        # dimensions are specified in the order of audio, video and text
        self.audio_hidden = hidden_dims[0]
        self.video_hidden = hidden_dims[1]
        self.text_hidden = hidden_dims[2]

        self.post_fusion_dim = post_fusion_dim

        self.audio_prob = dropouts[0]
        self.video_prob = dropouts[1]
        self.text_prob = dropouts[2]
        self.post_fusion_prob = dropouts[3]

        self.task = task

        # define the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        self.post_fusion_layer_1 = nn.Linear((self.text_hidden + 1) * (self.video_hidden + 1) * (self.audio_hidden + 1),
                                             self.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)

        # choose final layer according to task
        if self.task == 'binary':
            self.post_fusion_layer_3 = nn.Linear(self.post_fusion_dim, 1)
        elif sel.task == '5 class':
            self.post_fusion_layer_3 = nn.Linear(self.post_fusion_dim, 5)
        else:
            # regression task
            self.post_fusion_layer_3 = nn.Linear(self.post_fusion_dim, 1)
            # TFN regression with constrained output range: (-3, 3),
            # hence we'll apply sigmoid to output
            # shrink it to (0, 1), and scale\shift it back to range (-3, 3)
            self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
            self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, audio_x, video_x, text_x):
        '''
        Args:
            audio_x: tensor of shape (batch_size, audio_in)
            video_x: tensor of shape (batch_size, video_in)
            text_x: tensor of shape (batch_size, sequence_len, text_in)
        '''

        batch_size = audio_x.data.shape[0]

        # next we perform "tensor fusion", which is essentially appending 1s to the tensors and take Kronecker product
        if audio_x.is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor

        # concatenate representations with ones
        _audio_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), audio_x), dim=1)
        _video_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), video_x), dim=1)
        _text_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), text_x), dim=1)

        # _audio_h : (batch_size, audio_in + 1)
        # _video_h : (batch_size, _video_in + 1)
        # we want to perform batched outer product hence we unsqueenze them to get
        # (batch_size, audio_in + 1, 1) X (batch_size, 1, video_in + 1)
        # fusion_tensor will have shape (batch_size, audio_in + 1, video_in + 1)
        fusion_tensor = torch.bmm(_audio_h.unsqueeze(2), _video_h.unsqueeze(1))

        # next we do kronecker product between fusion_tensor and _text_h. This is even trickier
        # we have to reshape the fusion tensor during the computation
        # in the end we don't keep the 3-D tensor, instead we flatten it
        fusion_tensor = fusion_tensor.view(-1, (self.audio_hidden + 1) * (self.video_hidden + 1), 1)
        fusion_tensor = torch.bmm(fusion_tensor, _text_h.unsqueeze(1)).view(batch_size, -1)

        post_fusion_dropped = self.post_fusion_dropout(fusion_tensor)
        post_fusion_y_1 = F.relu(self.post_fusion_layer_1(post_fusion_dropped))
        post_fusion_y_2 = F.relu(self.post_fusion_layer_2(post_fusion_y_1))

        ### WE NEED TO FIX FORWARD TO PERFORM AT ALL GIVEN TASKS
        output = self.post_fusion_layer_3(post_fusion_y_2)

        # post_fusion_y_3 = F.sigmoid(self.post_fusion_layer_3(post_fusion_y_2))
        # output = post_fusion_y_3 * self.output_range + self.output_shift

        return output