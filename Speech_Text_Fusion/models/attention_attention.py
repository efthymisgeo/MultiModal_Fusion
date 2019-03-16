from torch import torch,nn

from config import BATCH_SIZE, MAX_LEN, DEVICE

from models.Text_Rnn import Text_RNN
from models.Fused_Attention import Attn_Fusion
from models.TensorFusionNetwork import TFN

class Hierarchy_Attn(nn.Module):

    def __init__(self, text_params, audio_params, post_fusion_dropout=0.1,
                 post_tfn_subnet=128):
        super(Hierarchy_Attn, self).__init__()

        # define text recurrent subnet
        self.text_rnn = Text_RNN(*text_params)

        # define audio recurrent subnet
        self.audio_rnn = Text_RNN(*audio_params)

        # define 1st attention level
        text_dim_tuple, audio_dim_tuple = \
            self.get_rnn_tuples(text_params[1], audio_params[1])
        self.fusion_net = Attn_Fusion(text_dim_tuple,
                                      audio_dim_tuple)
        # get text hidden size and audio hidden size
        H = text_dim_tuple[2]
        D = audio_dim_tuple[2]

        # define dense layers
        self.lin_1 = nn.Linear(H+2*D, D)
        self.lin_2 = nn.Linear(D, D)

        # define activation layers
        self.activ_1 = nn.ReLU()
        self.activ_2 = nn.ReLU()

        # map according to task
        self.lin_3 = nn.Linear(D,1)



    @staticmethod
    def get_rnn_tuples(hidden_text, hidden_audio):
        # 2 * comes from bidirectional
        text_tuple = (BATCH_SIZE, MAX_LEN, hidden_text*2)
        audio_tuple = (BATCH_SIZE, MAX_LEN, hidden_audio*2 )
        return text_tuple, audio_tuple


    def forward(self, covarep, glove, lengths):

        # sorted features accepted as input

        # text rnn
        _, T, hidden_t, weighted_t = self.text_rnn(glove, lengths)

        # audio rnn
        _, A, hidden_a, weighted_a = self.audio_rnn(covarep, lengths)

        # fused attention subnetwork
        F, weighted_fusion = self.fusion_net(hidden_t, weighted_t, hidden_a,
                                             weighted_a,lengths)
        # concatenate features
        fused_tensor = torch.cat((T, F, A), 1)

        # dense layers
        representations = self.activ_1(self.lin_1(fused_tensor))
        representations = self.activ_2(self.lin_2(representations))

        # project to task space
        logits = self.lin_3(representations)

        return logits