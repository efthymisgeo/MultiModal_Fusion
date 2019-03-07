from torch import nn, torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RNNEncoder(nn.Module):
    def __init__(self, input_size, rnn_size, num_layers,
                 bidirectional, dropout, architecture='LSTM'):
        """
        A simple RNN Encoder.
        Args:
            input_size (int): the size of the input features
            rnn_size (int): hidden state size
            num_layers (int):
            bidirectional (bool):
            dropout (float):
            architecture (str): LSTM or GRU
        Returns: outputs, last_outputs
        - **outputs** of shape `(batch, seq_len, hidden_size)`:
          tensor containing the output features `(h_t)`
          from the last layer of the LSTM, for each t.
        - **last_outputs** of shape `(batch, hidden_size)`:
          tensor containing the last output features
          from the last layer of the LSTM, for each t=seq_len.
        """
        super(RNNEncoder, self).__init__()

        # choose which RNN to use
        if architecture == 'LSTM':
            self.rnn = nn.LSTM(input_size=input_size,
                               hidden_size=rnn_size,
                               num_layers=num_layers,
                               bidirectional=bidirectional,
                               dropout=dropout,
                               batch_first=True)
        else:
            self.rnn = nn.GRU(input_size=input_size,
                              hidden_size=rnn_size,
                              num_layers=num_layers,
                              bidirectional=bidirectional,
                              dropout=dropout,
                              batch_first=True)


        # the dropout "layer" for the output of the RNN
        self.drop_rnn = nn.Dropout(dropout)

        # define output feature size
        self.feature_size = rnn_size

        if bidirectional:
            self.feature_size *= 2

    @staticmethod
    def last_by_index(outputs, lengths):
        # Index of the last output for each sequence.
        idx = (lengths - 1).view(-1, 1).expand(outputs.size(0),
                                               outputs.size(2)).unsqueeze(1)
        return outputs.gather(1, idx).squeeze()

    @staticmethod
    def split_directions(outputs):
        direction_size = int(outputs.size(-1) / 2)
        forward = outputs[:, :, :direction_size]
        backward = outputs[:, :, direction_size:]
        return forward, backward

    def last_timestep(self, outputs, lengths, bi=False):
        if bi:
            forward, backward = self.split_directions(outputs)
            last_forward = self.last_by_index(forward, lengths)
            last_backward = backward[:, 0, :]
            return torch.cat((last_forward, last_backward), dim=-1)

        else:
            return self.last_by_index(outputs, lengths)

    def forward(self, embs, lengths):
        """
        This is the heart of the model. This function, defines how the data
        passes through the network.
        Args:
            embs (): word embeddings
            lengths (): the lengths of each sentence
        Returns: the logits for each class
        """
        # pack the batch
        packed = pack_padded_sequence(embs, list(lengths.data),
                                      batch_first=True)

        out_packed, _ = self.rnn(packed)

        # unpack output - no need if we are going to use only the last outputs
        outputs, _ = pad_packed_sequence(out_packed, batch_first=True)

        # get the outputs from the last *non-masked* timestep for each sentence
        #last_outputs = self.last_timestep(outputs, lengths,
        #                                  self.rnn.bidirectional)

        # apply dropout to the outputs of the RNN
        last_outputs = self.drop_rnn(outputs)

        return outputs, last_outputs