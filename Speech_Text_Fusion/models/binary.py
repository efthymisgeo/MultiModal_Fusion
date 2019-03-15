from torch import torch,nn

from config import BATCH_SIZE, MAX_LEN, DEVICE

from models.Text_Rnn import Text_RNN
from models.FineTuningFusion import FTA_Fusion
from models.TensorFusionNetwork import TFN

class Bin_Model(nn.Module):

    def __init__(self, text_params, audio_params, text_rnn_path,
                 audio_rnn_path, post_fusion_dropout=0.1,
                 post_tfn_subnet=128):
        super(Bin_Model, self).__init__()

        self.text_rnn = self.load_pretrained_rnn(text_params, text_rnn_path)
        self.audio_rnn = self.load_pretrained_rnn(audio_params, audio_rnn_path)

        # get text/audio hidden sizes [1] and dropouts [4]
        hidden_dims, audio_drop, text_drop = \
            self.get_hidden_dropout(text_params, audio_params)

        dropouts = (audio_drop, audio_drop, text_drop, post_fusion_dropout)

        # FTAF Subnet definition
        text_dim_tuple, audio_dim_tuple = \
            self.get_rnn_tuples(text_params[1], audio_params[1])
        self.fusion_net = FTA_Fusion(text_dim_tuple, audio_dim_tuple)

        # TFN Network definition
        self.tfn = TFN(hidden_dims, dropouts, post_tfn_subnet)

    @staticmethod
    def get_hidden_dropout(text_params, audio_params):
        text_h = text_params[1]
        text_dropout = text_params[4]
        audio_h = audio_params[1]
        audio_dropout = audio_params[4]

        return (audio_h, audio_h, text_h), audio_dropout, text_dropout

    @staticmethod
    def get_rnn_tuples(hidden_text, hidden_audio):
        # 2 * comes from bidirectional
        text_tuple = (BATCH_SIZE, MAX_LEN, hidden_text*2)
        audio_tuple = (BATCH_SIZE, MAX_LEN, hidden_audio*2 )
        return text_tuple, audio_tuple

    @staticmethod
    def load_pretrained_rnn(rnn_params, rnn_path):
        # load pretrained rnn
        rnn_net = Text_RNN(*rnn_params)
        rnn_net.load_state_dict(torch.load(rnn_path))
        rnn_net.eval()

        # freeze weigths
        for param in rnn_net.parameters():
            param.requires_grad = False

        return(rnn_net)

    def forward(self, audio_batch, audio_lengths, text_batch, text_lengths):

        # audio_batch is [B,50,74]
        # text_batch is [B,50,300]

        # get batch representation, all hidden states and attentions
        _, A, audio_h, audio_w = self.audio_rnn(audio_batch, audio_lengths)
        _, T, text_h, text_w = self.text_rnn(text_batch, text_lengths)

        # fuse audio with text hidden states and attentions
        V, _ = self.fusion_net(text_h, text_w, audio_h, audio_w, _)

        # fuse three extracted batch representations
        logit = self.tfn(A, V, T)

        return(logit)