from config import DEVICE, MAX_LEN, BATCH_SIZE

###########################################
## Multiplication Attention Mechanism
## Multiplies both representations with other
## already extracted attention weights
###########################################


from torch import nn, torch
from torch.autograd import Variable



class SelfAttention(nn.Module):
    def __init__(self, attention_size,
                 batch_first=False,
                 layers=1,
                 dropout=0.15,
                 non_linearity="tanh"):
        super(SelfAttention, self).__init__()

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

    @staticmethod
    def get_mask(attentions, lengths):
        """
        Construct mask for padded itemsteps, based on lengths
        """
        max_len = max(lengths.data)
        mask = Variable(torch.ones(attentions.size())).detach()
        mask = mask.to(DEVICE)

        for i, l in enumerate(lengths.data):  # skip the first sentence
            if l < max_len:
                mask[i, l:] = 0
        return mask

    def forward(self, inputs, lengths):

        ##################################################################
        # STEP 1 - perform dot product
        # of the attention vector and each hidden state
        ##################################################################

        # inputs is a 3D Tensor: batch, len, hidden_size
        # scores is a 2D Tensor: batch, len
        scores = self.attention(inputs).squeeze()
        scores = self.softmax(scores)

        ##################################################################
        # Step 2 - Masking
        ##################################################################

        # construct a mask, based on sentence lengths
        mask = self.get_mask(scores, lengths)

        # apply the mask - zero out masked timesteps
        masked_scores = scores * mask

        # re-normalize the masked scores
        _sums = masked_scores.sum(-1, keepdim=True)  # sums per row
        scores = masked_scores.div(_sums)  # divide by row sum

        ##################################################################
        # Step 3 - Weighted sum of hidden states, by the attention scores
        ##################################################################

        # multiply each hidden state with the attention weights
        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))

        # sum the hidden states

        # in case of bug uncomment the following line
        # representations = weighted.sum(1).squeeze()
        representations = weighted.sum(1)

        return representations, scores



#################################################################
##### Fine Tuning Attention Fusion Mechanism ####################
#################################################################

class Mul_Fusion(nn.Module):
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

        super(Mul_Fusion, self).__init__()

        # get batch size, max_len, hidden_size
        B, K, H_t = text_size
        _, _, H_a = audio_size

        D = min(H_t, H_a)  # should be H_a
        H = max(H_t, H_a)  # should be H_t

        # mapping H --> D
        self.dense = nn.Linear(H, D)
        # generalized attention module
        self.attn = SelfAttention(D, batch_first=True, layers=layers)

    @staticmethod
    def weighted_timestep(hidden, weights):
        weighted_h = torch.mul(hidden,
                               weights.unsqueeze(-1).expand_as(hidden))
        return weighted_h

    @staticmethod
    def pad_mul(audio_h, text_h):
        pad_size = text_h.size(2) - audio_h.size(2)
        real_len = text_h.size(1)
        batch_size = text_h.size(0)

        ones = Variable(torch.ones(batch_size,
                                   real_len,
                                   pad_size)).detach()
        ones = ones.to(DEVICE)
        audio_h = torch.cat((audio_h, ones), 2)

        return(torch.mul(audio_h, text_h))

    def forward(self, h_text, w_text, h_audio, w_audio, lengths):
        '''
        INPUTS:
            h_text: [B,M,H]
            w_text: [B,M]
            h_audio: [B,M,D]
            w_audio: [B,M]
            lengths: [B,M]
        OUTPUTS:
        '''

        # get weighted representations
        #text_weighted = self.weighted_timestep(h_text, w_text)
        #audio_weighted = self.weighted_timestep(h_audio, w_audio)

        # Hadamard Product
        mul_fused = self.pad_mul(h_audio, h_text)

        ######################################################
        ## linear projection
        #mul_fused = self.dense(mul_fused)

        ## apply generalized attention
        #fused_representation, w_fusion = self.attn(mul_fused, lengths)

        return mul_fused