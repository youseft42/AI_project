import re
import torch
import torch.nn as nn
import torch.utils.data
from torch import Tensor
from typing import Iterator

class SubjectivityLSTM(nn.Module):
    def __init__(self, vocab_dim, embedding_dim, h_dim, n_layers=1, dropout=0):
        """
        :param in_dim: Number of input dimensions.
        :param h_dim: Number of hidden state dimensions.
        :param out_dim: Number of input dimensions.
        :param n_layers: Number of layer in the model.
        :param dropout: Level of dropout to apply between layers. Zero
        disables.
        """
        super().__init__()
        assert vocab_dim > 0 and h_dim > 0 and n_layers > 0

        # the first phase:
        self.embedding = nn.Embedding(vocab_dim, embedding_dim)

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=h_dim, num_layers=n_layers, dropout=dropout)

        self.linear = nn.Linear(h_dim, 3)

    def forward(self, x, **kw):
        # x shape: (S, B) Note batch dim is not first!
        embedded = self.embedding(x)  # embedded shape: (S, B, E)

        # GRU first  output: all hidden states from last layer (S, B, H)
        # GRU second output: last hidden state from each layer (L, B, H)
        h, ht = self.lstm(embedded)
        out = self.linear(h)

        return out, h, ht


class StanceLSTM(nn.Module):
    def __init__(self, vocab_dim, embedding_dim, h_dim, n_layers=1, dropout=0):
        """
        :param in_dim: Number of input dimensions.
        :param h_dim: Number of hidden state dimensions.
        :param out_dim: Number of input dimensions.
        :param n_layers: Number of layer in the model.
        :param dropout: Level of dropout to apply between layers. Zero
        disables.
        """
        super().__init__()
        assert vocab_dim > 0 and h_dim > 0 and n_layers > 0

        # the second phase:
        self.embedding = nn.Embedding(vocab_dim, embedding_dim)

        self.attn = torch.nn.MultiheadAttention(h_dim, h_dim)

        self.lstm = nn.LSTM(input_size=embedding_dim + h_dim*n_layers, hidden_size=h_dim, num_layers=n_layers, dropout=dropout)

        self.linear = nn.Linear(h_dim, 3)

    def forward(self, x, h_prev, sub_h, **kw):
        S, B = x.shape
        embedded = self.embedding(x)  # embedded shape: (S, B, E)

        q = h_prev[0]# (L,B, H)
        kv = sub_h# (S, B, H)
        a, _ = self.attn(q, kv, kv)  # (L, B , H)

        a = a.reshape((1, B, -1)).expand(S, -1, -1)
        rnn_input = torch.cat((embedded, a), dim=2)  # (S, B, E + L*H)

        # h:  (S, B, H)
        # ht: (L, B, H)
        h, ht = self.lstm(rnn_input, h_prev)

        # Project H back to the vocab size V, to get a score per word
        out = self.linear(h)

        # Out shapes: (S, B, V) and (L, B, H)
        return out, ht


class TwoPhaseLSTM(nn.Module):

    """
    Represents a two-phase LSTM model.
    """

    def __init__(self, subLSTM: SubjectivityLSTM, stanceLSTM: StanceLSTM):
        super().__init__()
        self.sub = subLSTM
        self.stance = stanceLSTM
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, x_src, **kw):

        # context is (L, B, H)
        sub_output, h, context = self.sub(x_src, **kw)
        stance_output, context = self.stance(x_src, context, h, **kw)
        y_stance = self.log_softmax(stance_output)
        y_sub = self.log_softmax(sub_output)
        # Output shape: (S-1, B, V)
        return y_stance, y_sub
