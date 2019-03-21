from torch import torch,nn

from config import BATCH_SIZE, MAX_LEN, DEVICE

from models.Text_Rnn import Text_RNN
from models.Fused_Attention import Attn_Fusion
from models.Mul_Attn import Mul_Fusion
from models.TensorFusionNetwork import TFN
from torch.autograd import Variable

class Hierarchy_Attn(nn.Module):

    def __init__(self, text_params, audio_params, fusion_params,
                 p_drop=0.15, post_tfn_subnet=128):
        super(Hierarchy_Attn, self).__init__()

        # define text recurrent subnet
        self.text_rnn = Text_RNN(*text_params)

        # define audio recurrent subnet
        self.audio_rnn = Text_RNN(*audio_params)

        # define fusion RNN net
        self.fusion_rnn = Text_RNN(*fusion_params)

        # define cat-fusion attention level
        text_dim_tuple, audio_dim_tuple = \
            self.get_rnn_tuples(text_params[1], audio_params[1])
        self.fusion_net = Attn_Fusion(text_dim_tuple,
                                      audio_dim_tuple)

        # define mul-fusion layer
        self.mul_fusion = Mul_Fusion(text_dim_tuple,
                                     audio_dim_tuple)

        # get text hidden size and audio hidden size
        H = text_dim_tuple[2]
        D = audio_dim_tuple[2]

        # fused reps dimensionality reduction
        self.fusion_transform = nn.Sequential(nn.Linear(2*H+D, H),
                                              nn.Dropout(p_drop),
                                              nn.ReLU(),
                                              nn.Linear(H, H//2),
                                              nn.Dropout(p_drop),
                                              nn.ReLU())

        # deep representations
        self.deep_audio = nn.Sequential(nn.Linear(D,D),
                                        nn.Dropout(p_drop),
                                        nn.ReLU(),
                                        nn.Linear(D,D),
                                        nn.Dropout(p_drop),
                                        nn.ReLU())

        self.deep_text = nn.Sequential(nn.Linear(H,H),
                                       nn.ReLU(),
                                       nn.Linear(H,H),
                                       nn.ReLU())
        fusion_dim = fusion_params[1]*2
        self.deep_fused = nn.Sequential(nn.Linear(fusion_dim,fusion_dim),
                                        nn.Dropout(p_drop),
                                        nn.ReLU(),
                                        nn.Linear(fusion_dim, fusion_dim),
                                        nn.Dropout(p_drop),
                                        nn.ReLU())

        ################################
        ## deep feature + reps networks
        ###############################
        self.deep_audio_2 = nn.Sequential(nn.Linear(2*D,D),
                                          nn.Dropout(p_drop),
                                          nn.ReLU())

        self.deep_text_2 = nn.Sequential(nn.Linear(2*H,H),
                                         nn.ReLU())

        self.deep_fusion_2 = nn.Sequential(nn.Linear(2*D,D),
                                           nn.Dropout(p_drop),
                                           nn.ReLU())

        ###################################
        ### final fusion layer
        ###################################

        self.deep_mul = nn.Sequential(nn.Linear(D,D),
                                      nn.Dropout(p_drop),
                                        nn.ReLU(),
                                        nn.Linear(D,D),
                                      nn.Dropout(p_drop),
                                      nn.ReLU())

        # define representation layers
        self.audio_fused_net = nn.Linear(2*D, D)

        self.text_fused_net = nn.Sequential(nn.Linear(H+D,H//2),
                                            nn.Linear(H//2, D))

        self.text_mul_net = nn.Sequential(nn.Linear(H+D, H),
                                          nn.Linear(H, H//2),
                                          nn.Linear(H//2 ,D))

        self.audio_mul_net = nn.Sequential(nn.Linear(2*D, D),
                                           nn.Linear(D, D))

        self.cat_audio_net = nn.Linear(2*D, D)

        self.cat_text_net = nn.Sequential(nn.Linear(H+D, H//2),
                                          nn.Linear(H//2, D))

        self.mul_cat_net = nn.Sequential(nn.Linear(H+3*D,H),
                                         nn.Linear(H,D),
                                         nn.Linear(D, D))

        self.deep_cat_net = nn.Sequential(nn.Linear(2*D,D),
                                          nn.Linear(D,D))

        self.deep_mul_net = nn.Sequential(nn.Linear(2*D,D),
                                          nn.Linear(D,D))

        # define dense layers
        cat_size = 2*H+3*D
        self.dense = nn.Sequential(nn.Linear(cat_size,cat_size//2),
                                   nn.Dropout(p_drop),
                                   nn.ReLU(),
                                   nn.Linear(cat_size//2,H),
                                   nn.Dropout(p_drop),
                                   nn.ReLU(),
                                   nn.Linear(H, D),
                                   nn.Dropout(p_drop),
                                   nn.ReLU(),
                                   nn.Linear(D,D),
                                   nn.Dropout(p_drop),
                                   nn.ReLU())

        # modality attention layers

        ##############################################
        ## h,g,f,m with tf,tm - af,am - tf,af - tm,am
        ##############################################
        self.audio_attn = nn.Sequential(nn.Linear(2*D, D),
                                        nn.Linear(D, D//2),
                                        nn.Linear(D//2, 1))
        self.text_attn = nn.Sequential(nn.Linear(2 * D, D),
                                       nn.Linear(D, D//2),
                                       nn.Linear(D//2, 1))
        self.fused_attn = nn.Sequential(nn.Linear(2 * D, D),
                                       nn.Linear(D, D // 2),
                                       nn.Linear(D // 2, 1))
        self.mul_attn = nn.Sequential(nn.Linear(2 * D, D),
                                       nn.Linear(D, D // 2),
                                       nn.Linear(D // 2, 1))

        '''
        self.text_attn = nn.Sequential(nn.Linear(H, H//2),
                                       nn.Linear(H//2, H//4),
                                       nn.Linear(H//4, 1))
        
        self.fused_attn = nn.Sequential(nn.Linear(D, D//2),
                                        nn.Linear(D//2, 1))

        self.mul_attn = nn.Sequential(nn.Linear(D, D//2),
                                      nn.Linear(D//2, 1))

        self.a_fused_attn = nn.Sequential(nn.Linear(D, D//2),
                                          nn.Linear(D//2, 1))

        self.t_fused_attn = nn.Sequential(nn.Linear(D, D//2),
                                          nn.Linear(D//2, 1))

        self.a_mul_attn = nn.Sequential(nn.Linear(D, D//2),
                                        nn.Linear(D//2, 1))

        self.t_mul_attn = nn.Sequential(nn.Linear(D, D//2),
                                        nn.Linear(D//2, 1))

        self.mul_cat_attn = nn.Sequential(nn.Linear(D, D//2),
                                          nn.Linear(D//2, 1))

        self.deep_cat_attn = nn.Sequential(nn.Linear(D, D//2),
                                           nn.Linear(D//2, 1))

        self.deep_mul_attn = nn.Sequential(nn.Linear(D, D//2),
                                           nn.Linear(D//2, 1))

        '''
        self.softmax = nn.Softmax(dim=-1)

        # map according to task
        self.fusion_mapping = nn.Linear(D,1)
        self.audio_mapping = nn.Linear(D,1)
        self.text_mapping = nn.Linear(H,1)

    @staticmethod
    def zero_pad(tensor):
        batch_size = tensor.size(0)
        real_len = tensor.size(1)
        dim = tensor.size(2)

        if MAX_LEN > real_len:
            zeros = \
                Variable(torch.zeros(batch_size,
                                    MAX_LEN-real_len,
                                    dim)).detach()
            zeros = zeros.to(DEVICE)

            tensor = torch.cat((tensor, zeros), 1)

        return(tensor)

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

        # cat-fusion attention subnetwork
        f_i = self.fusion_net(hidden_t, weighted_t,
                              hidden_a, weighted_a, lengths)

        # mul-fusion attention network
        m_i = self.mul_fusion(hidden_t, weighted_t, hidden_a,
                              weighted_a, lengths)

        fused_i = torch.cat((f_i, m_i), 2)
        fused_i = self.zero_pad(fused_i)

        fused_i = self.fusion_transform(fused_i)

        _, F, _, _ = self.fusion_rnn(fused_i, lengths)


        # dense representations
        deep_A_ = self.deep_audio(A)
        deep_T_ = self.deep_text(T)
        deep_F_ = self.deep_fused(F)
        #M = self.deep_mul(M)

        # concatenate features
        deep_A = torch.cat((A, deep_A_), 1)
        deep_T = torch.cat((T, deep_T_), 1)
        deep_F = torch.cat((F, deep_F_), 1)

        # extract generalized features
        deep_A = self.deep_audio_2(deep_A)
        deep_T = self.deep_text_2(deep_T)
        deep_F = self.deep_fusion_2(deep_F)

        # final feature list
        representations_list = [A, T, deep_A, deep_T, deep_F]# deep_A, deep_T, deep_F]


        '''
        # concatenate features
        # audio-fused
        A_F = torch.cat((A, F), 1)
        # text-fused
        T_F = torch.cat((T, F), 1)
        # audio-mul
        A_M = torch.cat((A, M), 1)
        # text-mul
        T_M = torch.cat((T, M), 1)
        # cat-mul
        #F_M = torch.cat((A, T, F, M), 1)

        # extract fused representations
        #deep_fm = self.mul_cat_net(F_M)
        
        deep_af = self.audio_fused_net(A_F)
        deep_tf = self.text_fused_net(T_F)

        deep_am = self.audio_mul_net(A_M)
        deep_tm = self.text_mul_net(T_M)

        a_attn = torch.cat((deep_af, deep_am), 1)
        t_attn = torch.cat((deep_tf, deep_tm), 1)
        f_attn = torch.cat((deep_tf, deep_tm), 1)
        m_attn = torch.cat((deep_af, deep_am), 1)

        # cat deep fused representations
        #AF_TF = torch.cat((deep_af, deep_tf), 1)
        #AM_TM = torch.cat((deep_am, deep_tm), 1)

        # extract deep fused representations
        #deep_cat = self.deep_cat_net(AF_TF)
        #deep_mul = self.deep_mul_net(AM_TM)


        # cat all extracted representations

        

        representations_list = [A,T,F,M]


         #                       deep_fm,
         #                       deep_cat, deep_mul]
        #representations_list = [A, T, F, M, deep_fm, deep_af,
        #                        deep_tf, deep_am, deep_tm]

        # modality attention
        audio_energy = self.audio_attn(a_attn)
        text_energy = self.text_attn(t_attn)
        fused_energy = self.fused_attn(f_attn)
        mul_energy = self.mul_attn(m_attn)

        #fm_energy = self.mul_cat_attn(deep_fm)

        # VERSION 3c
        
        a_fused_energy = self.a_fused_attn(deep_af)
        t_fused_energy = self.t_fused_attn(deep_tf)

        a_mul_energy = self.a_mul_attn(deep_am)
        t_mul_energy = self.t_mul_attn(deep_tm)
        
        #deep_cat_energy = self.deep_cat_attn(deep_cat)
        #deep_mul_energy = self.deep_mul_attn(deep_mul)

        # version 3c
        
        energies_list = [a_fused_energy, t_fused_energy,
                         a_mul_energy, t_mul_energy]
    

        energies_list = [audio_energy, text_energy,
                         fused_energy, mul_energy]


        energies = torch.cat(energies_list, 1)
        energies = self.softmax(energies)

        for idx, rep in enumerate(representations_list):
            representations_list[idx] = \
                torch.mul(rep,
                          energies[:,idx].unsqueeze(-1).expand_as(rep))

        '''
        # concatenate all existing representations
        deep_representations = torch.cat(representations_list, 1)

        # dense layers
        representations = self.dense(deep_representations)

        # project to task space
        logits_fusion = self.fusion_mapping(representations)
        logits_audio = self.audio_mapping(deep_A)
        logits_text = self.text_mapping(deep_T)

        return logits_fusion, logits_audio, logits_text