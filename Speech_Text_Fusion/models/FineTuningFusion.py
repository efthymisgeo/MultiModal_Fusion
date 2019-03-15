from torch import nn, torch
from torch.autograd import Variable
from config import DEVICE, MAX_LEN

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

        # sum the hidden states
        # uncomment following command in case of bug
        # representations = weighted.sum(1).squeeze()
        representations = weighted.sum(1)

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
            text_size: (B,K,H_t)
            audio_size: (B,K,H_a)
            layers (int): # of attn layers
            fusion_size (int)
            attention_size (int)
        OUTPUTS: (from forward)
            fused: (B,min{H_t,H_a}) fused representation
            fused_a : (B,K) fusion attentions
        '''

        super(FTA_Fusion, self).__init__()

        # get batch size, max_len, hidden_size
        B, K, H_t = text_size
        _, _, H_a = audio_size

        W = min(H_t, H_a)  # should be H_a

        # mapping (H_t + H_a)*MAX_LEN --> W*MAX_LEN
        self.dense = nn.Linear(H_t+H_a, W)
        # generalized attention module
        self.attn = GeneralAttention(W, batch_first=True, layers=layers)



    def forward(self, h_text, w_text, h_audio, w_audio, lengths):
        '''
        INPUTS:
            h_text:
            w_text:
            h_audio:
            w_audio:
            lengths:
        OUTPUTS:
        '''

        # cat features
        h_fused = torch.cat((h_text, h_audio), 2)

        # linear projection
        h_fused = self.dense(h_fused)

        # average attention energies
        w_averaged = torch.add(w_text, w_audio) / 2.0

        # apply generalized attention
        fusion_representation, w_fusion = self.attn(h_fused, w_averaged)

        return fusion_representation, w_fusion