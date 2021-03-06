from torch import nn, torch
from models.BiRNN import RNNEncoder
from models.Attention import SelfAttention

class Text_RNN(nn.Module):
    def __init__(self, input_size, rnn_size, num_layers,
                 bidirectional, dropout, architecture,
                 attention_size, batch_first, attn_layers,
                 attn_dropout, attn_nonlinearity,
                 task):
        '''
        Our Text RNN which functions on word level
        Uses RNNEncoder and SelfAttention
        Args:
            input_size (int): input features
            rnn_size (int): hidden size
            num_layers (int):
            bidirectional(bool):
            dropout (float): rnn dropout
            architecture (str): LSTM or GRU
            attention_size (int): attention size
            batch_first (bool):
            attn_layers (int):
            attn_dropout (float):
            attn_nonlinearity (str):
        Outputs:
            returns 3 tensors
            logits [B,*] : *=1 for binary, *=5 for 5class
            attn_representations (all hidden states) [B,50,128]
            attn_scores (weights) [B,50]
        '''

        super(Text_RNN, self).__init__()

        self.rnn = RNNEncoder(input_size=input_size,
                              rnn_size=rnn_size,
                              num_layers=num_layers,
                              bidirectional=bidirectional,
                              dropout=dropout,
                              architecture=architecture)

        if bidirectional == True:
            attention_size = 2*attention_size

        self.attn = SelfAttention(attention_size=attention_size,
                                  batch_first=batch_first,
                                  layers=attn_layers,
                                  dropout=attn_dropout,
                                  non_linearity=attn_nonlinearity)

        if task == 'Binary':
            output_size = 1
        else:
            # 5 class clf
            output_size = 5


        self.dense = nn.Linear(attention_size, output_size)
        self.activ = nn.ReLU()
        ###maybe needs fix because sequential needs
        ##list as inputs

    def forward(self, inputs, lengths):
        '''
        Input Args:
            inputs: 3D tensor (batch_len, max_len, feature_dim)
            lengths: 1D tensor (batch_len)
        Output Args:
            logits: (batch_len, *)
            attn_representations: (batch_len, hidden_dim)
            rnn_out (batch_len, real_max_len, hidden_dim)
            attn_scores: (batch_len, real_max_len)
        '''

        # Pass batch input through RNN #
        #       Get Packed outputs     #
        rnn_out, last_outputs = self.rnn(inputs, lengths)
       
        #  Apply Attention in rnn_out  #
        attn_representations, attn_scores = self.attn(rnn_out,
                                                      lengths)
        # project to #classes
        logits = self.dense(attn_representations).view(-1,1)

        return logits, attn_representations, rnn_out, attn_scores


class Text_Encoder(nn.Module):
    def __init__(self, input_size, rnn_size, num_layers,
                 bidirectional, dropout, architecture,
                 attention_size, batch_first, attn_layers,
                 attn_dropout, attn_nonlinearity,
                 task):
        '''
        Our Text RNN which functions on word level
        Uses RNNEncoder and SelfAttention
        Args:
            input_size (int): input features
            rnn_size (int): hidden size
            num_layers (int):
            bidirectional(bool):
            dropout (float): rnn dropout
            architecture (str): LSTM or GRU
            attention_size (int): attention size
            batch_first (bool):
            attn_layers (int):
            attn_dropout (float):
            attn_nonlinearity (str):
        Outputs:
            returns 3 tensors
            logits [B,*] : *=1 for binary, *=5 for 5class
            attn_representations (all hidden states) [B,50,128]
            attn_scores (weights) [B,50]
        '''

        super(Text_Encoder, self).__init__()

        self.rnn = RNNEncoder(input_size=input_size,
                              rnn_size=rnn_size,
                              num_layers=num_layers,
                              bidirectional=bidirectional,
                              dropout=dropout,
                              architecture=architecture)

        if bidirectional == True:
            attention_size = 2 * attention_size

        self.attn = SelfAttention(attention_size=attention_size,
                                  batch_first=batch_first,
                                  layers=attn_layers,
                                  dropout=attn_dropout,
                                  non_linearity=attn_nonlinearity)

        if task == 'Binary':
            output_size = 1
        else:
            # 5 class clf
            output_size = 5

        H = attention_size
        self.deep_text = nn.Sequential(nn.Linear(H,H),
                                       nn.ReLU(),
                                       nn.Linear(H,H),
                                       nn.ReLU())

        self.dense = nn.Linear(attention_size, output_size)

        self.activ = nn.ReLU()
        ###maybe needs fix because sequential needs
        ##list as inputs

    def forward(self, inputs, lengths):
        '''
        Input Args:
            inputs: 3D tensor (batch_len, max_len, feature_dim)
            lengths: 1D tensor (batch_len)
        Output Args:
            logits: (batch_len, *)
            attn_representations: (batch_len, hidden_dim)
            rnn_out (batch_len, real_max_len, hidden_dim)
            attn_scores: (batch_len, real_max_len)
        '''

        # Pass batch input through RNN #
        #       Get Packed outputs     #
        rnn_out, last_outputs = self.rnn(inputs, lengths)

        #  Apply Attention in rnn_out  #
        attn_representations, attn_scores = self.attn(rnn_out,
                                                      lengths)
        # project to #classes
        representations = self.deep_text(attn_representations)
        logits = self.dense(representations).view(-1, 1)

        return logits, representations, rnn_out, attn_scores, attn_representations


class Audio_Encoder(nn.Module):
    def __init__(self, input_size, rnn_size, num_layers,
                 bidirectional, dropout, architecture,
                 attention_size, batch_first, attn_layers,
                 attn_dropout, attn_nonlinearity,
                 task):
        '''
        Our Text RNN which functions on word level
        Uses RNNEncoder and SelfAttention
        Args:
            input_size (int): input features
            rnn_size (int): hidden size
            num_layers (int):
            bidirectional(bool):
            dropout (float): rnn dropout
            architecture (str): LSTM or GRU
            attention_size (int): attention size
            batch_first (bool):
            attn_layers (int):
            attn_dropout (float):
            attn_nonlinearity (str):
        Outputs:
            returns 3 tensors
            logits [B,*] : *=1 for binary, *=5 for 5class
            attn_representations (all hidden states) [B,50,128]
            attn_scores (weights) [B,50]
        '''

        super(Audio_Encoder, self).__init__()

        self.rnn = RNNEncoder(input_size=input_size,
                              rnn_size=rnn_size,
                              num_layers=num_layers,
                              bidirectional=bidirectional,
                              dropout=dropout,
                              architecture=architecture)

        if bidirectional == True:
            attention_size = 2 * attention_size

        self.attn = SelfAttention(attention_size=attention_size,
                                  batch_first=batch_first,
                                  layers=attn_layers,
                                  dropout=attn_dropout,
                                  non_linearity=attn_nonlinearity)

        if task == 'Binary':
            output_size = 1
        else:
            # 5 class clf
            output_size = 5

        H = attention_size
        self.deep_text = nn.Sequential(nn.Linear(H,H),nn.Dropout(0.5),
                                       nn.ReLU(),
                                       nn.Linear(H,H),nn.Dropout(0.5),
                                       nn.ReLU())

        self.dense = nn.Linear(attention_size, output_size)

        self.activ = nn.ReLU()
        ###maybe needs fix because sequential needs
        ##list as inputs

    def forward(self, inputs, lengths):
        '''
        Input Args:
            inputs: 3D tensor (batch_len, max_len, feature_dim)
            lengths: 1D tensor (batch_len)
        Output Args:
            logits: (batch_len, *)
            attn_representations: (batch_len, hidden_dim)
            rnn_out (batch_len, real_max_len, hidden_dim)
            attn_scores: (batch_len, real_max_len)
        '''

        # Pass batch input through RNN #
        #       Get Packed outputs     #
        rnn_out, last_outputs = self.rnn(inputs, lengths)

        #  Apply Attention in rnn_out  #
        attn_representations, attn_scores = self.attn(rnn_out,
                                                      lengths)
        # project to #classes
        representations = self.deep_text(attn_representations)
        logits = self.dense(representations).view(-1, 1)

        return logits, representations, rnn_out, attn_scores, attn_representations
