from torch import torch, nn

from models.Attention import SelfAttention

class GeneralAttention(nn.Module):
    def __init__(self, attention_size, batch_first=False,
                 layers=1, dropout=.0, non_linearity='tanh'):
        '''
        generalization of classic Self Attention
        INPUTS:
            attention_size (int)
            batch_first (bool)
            layers (int)
            dropout [0,1]
            non_linearity (str)
        '''
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

    def forward(self, inputs, weigths, lengths):

        ##################################################################
        # STEP 1 - perform dot product
        # of the attention vector and each hidden state
        ##################################################################

        # inputs is a 2D Tensor: batch, hidden_size
        # scores is a 2D Tensor: batch, len
        scores = self.attention(inputs).squeeze()
        scores = self.softmax(scores)
        scores = scores +

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
        representations = weighted.sum(1).squeeze()

        return representations, scores

class FTAttnFusion(nn.Module):
    def __init__(self, text_h, text_a, audio_h, audio_a, layers):
        '''
        class abstraction that implements
        -- Fine Tuning Attention Mechanism --
        * : BATCH SIZE
        INPUTS:
            text_h: (*,H) text representation
            text_a: (*,H) text attn weights
            audio_h: (*,H__) audio representation
            audio_a: (*,H__) audio attn weights
            layers (int): # of attn layers
        OUTPUTS:
            fused: (*,max{H,H__}) fused representation
            fused_a : (same as above) fusion attentions
        '''

        super(FTAttnFusion, self).__init__()

        # get hidden sizes
        H = text_a.size(1)
        H__ = audio_a.size(1)

        # fused representation size
        W = max(H,H__)

        self.Dense = nn.Linear(H+H__,W)
        self.Attention = GeneralAttention(W, batch_first=True,
                                       layers=layers)
