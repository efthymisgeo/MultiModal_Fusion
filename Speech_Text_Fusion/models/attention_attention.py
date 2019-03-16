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

        # define representation layers
        self.audio_fused_net = nn.Linear(2*D, D)
        self.text_fused_net = nn.Linear(H+D, D)
        self.deep_fused_net = nn.Linear(2*D, D)

        # define dense layers
        self.lin_1 = nn.Linear(H+5*D, D)
        self.lin_2 = nn.Linear(D, D)

        # modality attention layers
        self.audio_attn = nn.Sequential(nn.Linear(D, D),
                                        nn.Linear(D, 1))
        self.text_attn = nn.Sequential(nn.Linear(H, D),
                                        nn.Linear(D, 1))
        self.fused_attn = nn.Sequential(nn.Linear(D, D),
                                        nn.Linear(D, 1))
        self.a_fused_attn = nn.Sequential(nn.Linear(D, D),
                                        nn.Linear(D, 1))
        self.t_fused_attn = nn.Sequential(nn.Linear(D, D),
                                        nn.Linear(D, 1))
        self.deep_fused_attn = nn.Sequential(nn.Linear(D, D),
                                        nn.Linear(D, 1))

        self.softmax = nn.Softmax(dim=-1)

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
        # audio-fused
        A_F = torch.cat((A, F), 1)
        # text-fused
        T_F = torch.cat((T, F), 1)

        # extract fused representations
        deep_af = self.audio_fused_net(A_F)
        deep_tf = self.text_fused_net(T_F)

        # extract deep fused representations
        AF_TF = torch.cat((deep_af, deep_tf), 1)

        deep_af_tf = self.deep_fused_net(AF_TF)

        representations_list = [A,T,F,deep_af,deep_tf,deep_af_tf]

        # modality attention
        audio_energy = self.audio_attn(A)
        text_energy = self.text_attn(T)
        fused_energy = self.fused_attn(F)
        a_fused_energy = self.a_fused_attn(deep_af)
        t_fused_energy = self.t_fused_attn(deep_tf)
        deep_fused_energy = self.deep_fused_attn(deep_af_tf)

        energies_list = [audio_energy, text_energy, fused_energy,
                         a_fused_energy, t_fused_energy, deep_fused_energy]

        energies = torch.cat(energies_list, 1)
        energies = self.softmax(energies)

        for idx, rep in enumerate(representations_list):
            representations_list[idx] = torch.mul(rep, energies[:,idx].unsqueeze(-1).expand_as(rep))


        # concatenate all existing representations
        deep_representation = torch.cat(representations_list, 1)

        # dense layers
        representations = self.activ_1(self.lin_1(deep_representation))
        representations = self.activ_2(self.lin_2(representations))

        # project to task space
        logits = self.lin_3(representations)

        return logits