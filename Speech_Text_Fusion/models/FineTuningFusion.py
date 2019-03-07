from torch import nn, torch
from torch.autograd import Variable
from config import DEVICE

###########################################
## Generalised Attention Mechanism  #######
## Fuses both representations and other
## already extracted attention weights
###########################################


class GeneralAttention(nn.Module):
    def __init__(self, attention_size, batch_first=True,
                 layers=1, dropout=.0, non_linearity="tanh"):
        super(GeneralAttention, self).__init__()

        self.batch_first = batch_first

        if non_linearity == "relu":
            activation = nn.ReLU()
        else:
            activation = nn.Tanh()

        modules = []
        for i in range(layers - 1):
            modules.append(nn.Linear(attention_size, attention_size))
            modules.append(activation)
            modules.append(nn.Dropout(dropout))

        # last attention layer must output 1
        modules.append(nn.Linear(attention_size, 1))
        modules.append(activation)
        modules.append(nn.Dropout(dropout))

        self.attention = nn.Sequential(*modules)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs, weigths):

        # inputs is a 3D Tensor: batch, len, hidden_size
        # weights is a 2D Tensor: batch, len
        # scores is a 2D Tensor: batch, len
        scores = self.attention(inputs).squeeze()
        scores = self.softmax(scores)
        scores = torch.add(scores, weigths)

        # re-normalize the fused scores
        _sums = scores.sum(-1, keepdim=True)  # sums per row
        scores = scores.div(_sums)  # divide by row sum

        # multiply each hidden state with the attention weights
        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))
        print(weighted.size())
        # sum the hidden states
        representations = weighted.sum(1).squeeze()
        print(representations.size())
        return representations, scores


#################################################################
##### Fine Tuning Attention Fusion Mechanism ####################
#################################################################

class FTA_Fusion(nn.Module):
    def __init__(self, text_size, audio_size, layers=1,
                 fusion_size=38, attention_size=68):
        '''
        class abstraction that implements
        -- Fine Tuning Attention Mechanism --
        B : BATCH SIZE
        INPUTS:
            text_size: (B,M,H_t)
            audio_size: (B,L,H_a)
            layers (int): # of attn layers
            fusion_size (int)
            attention_size (int)
        OUTPUTS:
            fused: (B,max{H_t,H_a}) fused representation
            fused_a : (B,max{M,L}) fusion attentions
        '''

        super(FTA_Fusion, self).__init__()

        # get batch size, max_len, hidden_size
        B, M, H_t = text_size
        _, L, H_a = audio_size

        W = min(H_t, H_a)  # should be H_a
        Q = max(M, L)  # should be L

        # mapping (H_t + H_a)*B --> W
        self.dense = nn.Linear((H_t + H_a) * Q, W * Q)
        # generalized attention module
        self.attn = GeneralAttention(W)

    @staticmethod
    def pad_cat(tensor_1, tensor_2):
        '''
        Args:
        tensor_1: (B,M,D)
        tensor_2: (B,L,H)
        Outputs:
        tensor: (B,max{M,L},D+H)
        '''
        B, M, D = tensor_1.size()
        _, L, H = tensor_2.size()
        # pad
        if M > L:
            pad_tensor = torch.zeros(B, M - L, H).to(DEVICE)
            tensor_2 = torch.cat((tensor_2, pad_tensor), 1)
        else:
            pad_tensor = torch.zeros(B, L - M, D).to(DEVICE)
            tensor_1 = torch.cat((tensor_1, pad_tensor), 1)
        # concatenate
        tensor = torch.cat((tensor_1, tensor_2), 2)
        return tensor

    @staticmethod
    def pad_mean(tensor_1, tensor_2):
        '''
        Args:
        tensor_1: (B,M)
        tensor_2: (B,L)
        Outputs:
        tensor: (B,max{M,L})
        '''
        B, M = tensor_1.size()
        _, L = tensor_2.size()
        # pad
        if M > L:
            pad_tensor = torch.zeros(B, M - L).to(DEVICE)
            tensor_2 = torch.cat((tensor_2, pad_tensor), 1)
        else:
            pad_tensor = torch.zeros(B, L - M).to(DEVICE)
            tensor_1 = torch.cat((tensor_1, pad_tensor), 1)
        # mean
        tensor = torch.add(tensor_1, tensor_2) / 2.0
        return tensor

    def forward(self, h_text, w_text, h_audio, w_audio):
        '''
        Args
        h_text (B,M,H_t) : hidden text representation
        w_text (B,M): text attn weigths
        h_audio (B,L,H_a) : hid audio representation
        w_audio (B,L): audio attn weights
        Outputs
        fused ()
        '''
        # get sizes
        B, M, H_t = h_text.size()
        _, L, H_a = h_audio.size()

        W = max(M, L)
        H = min(H_t, H_a)

        # concatenate text&audio representations
        joint_repr = self.pad_cat(h_text, h_audio).view(B, -1)
        print(joint_repr.size())
        joint_repr = self.dense(joint_repr).view(B, W, -1)

        # get mean of text, audio attention vectors
        joint_attn = self.pad_mean(w_text, w_audio)

        # attention layer
        joint_repr, joint_weights = self.attn(joint_repr, joint_attn)
        print(joint_repr.size())
        print(joint_weights.size())

        ##############################
        ## at this point we have (8,68,38) repr
        ## and (8,68) weights whereas we need
        ## (8,38) representation
        ##########################
        ## FIX ME
        # element-wise product
        # joint_repr = torch.mul(joint_repr,
        #                     joint_attn)

        return joint_repr, joint_weights

