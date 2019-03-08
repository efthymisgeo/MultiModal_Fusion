from torch import torch,nn

from models.Text_Rnn import Text_RNN
from models.FineTuningFusion import FTA_Fusion
from models.TensorFusionNetwork import TFN

class Bin_Model(nn.Module):

    def __init__(self): # fill parameters

        super(Bin_Model, self).__init__()

        # false initilization because we need to initiliaze
        # our pretrained text and audio RNNs
        self.text_net = Text_RNN() # fill parameters
        self.audio_net = Text_RNN() # fill parameters

        self.fusion_net = FTA_Fusion()

        self.tfn = TFN()





    def forward(self, audio_batch, text_batch):

        # audio_batch is [B,68,72]
        # text_batch is [B,50,300]

        # get batch representation, all hidden states and attentions
        A, audio_h, audio_w = self.audio_net(audio_batch)
        T, text_h, text_w = self.text_net(text_batch)

        # fuse audio with text hidden states and attentions
        V,_ = self.fusion_net(text_h, text_w, audio_h, audio_w)

        # fuse three extracted batch representations
        logit = self.tfn(A, V, T)

        return(logit)