import re
import torch
import torch.nn as nn
import torch.utils.data
import math
from torch import Tensor
from typing import Iterator

def create_emb_layer(weights_matrix):
    num_embeddings, embedding_dim = weights_matrix.size()
    weight = torch.FloatTensor(weights_matrix)
    emb_layer = nn.Embedding.from_pretrained(weight)
    # emb_layer.load_state_dict({'weight': weights_matrix})
    return emb_layer, num_embeddings, embedding_dim


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=80):
        super().__init__()
        self.d_model = d_model

        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        x = x.transpose(0, 1)
        seq_len = x.shape[1]
        x = x + torch.Tensor(self.pe[:, :seq_len, :])
        x = x.transpose(0, 1)
        # print(x.shape)
        return x

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        # We set d_ff as a default to 2048
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        x = self.dropout(x)
        x = torch.relu(self.linear_1(x))
        x = torch.relu(self.linear_2(x))
        return x

class SubjectivityLSTM(nn.Module):
    def __init__(self, vocab_dim, embedding_dim, h_dim, n_layers=1, dropout=0.1):
        """
        :param vocab_dim: Number of vocabulary dimensions.
        :param h_dim: Number of hidden state dimensions.
        :param embedding_dim: Number of embedding dimensions.
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

        embedded = self.embedding(x)

        h, ht = self.lstm(embedded)
        out = self.linear(h)

        return out, h, ht


class StanceLSTM(nn.Module):
    def __init__(self, vocab_dim, embedding_dim, h_dim, n_layers=1, dropout=0.1):
        """
        :param vocab_dim: Number of vocabulary dimensions.
        :param h_dim: Number of hidden state dimensions.
        :param embedding_dim: Number of embedding dimensions.
        :param n_layers: Number of layer in the model.
        :param dropout: Level of dropout to apply between layers. Zero
        disables.
        """
        super().__init__()
        assert vocab_dim > 0 and h_dim > 0 and n_layers > 0

        # the second phase:
        self.embedding = nn.Embedding(vocab_dim, embedding_dim)

        self.attn = torch.nn.MultiheadAttention(h_dim, 1)

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

        # Project H back to the output size to get prediction for the stance
        out = self.linear(h)

        # Out shapes: (S, B, V) and (L, B, H)
        return out, ht


class TwoPhaseLSTM(nn.Module):

    """
    Represents a two-phase LSTM model.
    """

    def __init__(self, vocab_dim, embedding_dim, h_dim, n_layers=1, dropout=0.1):
        super().__init__()
        self.sub = SubjectivityLSTM(vocab_dim, embedding_dim, h_dim, n_layers, dropout)
        self.stance = StanceLSTM(vocab_dim, embedding_dim, h_dim, n_layers, dropout)
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, x_src, **kw):

        # context is (L, B, H)
        sub_output, h, context = self.sub(x_src, **kw)
        stance_output, context = self.stance(x_src, context, h, **kw)
        y_stance = self.log_softmax(stance_output)
        y_sub = self.log_softmax(sub_output)
        # Output shape: (S, B, 3)
        return y_stance, y_sub

class PeSubjectivityLSTM(nn.Module):
    def __init__(self, weights_matrix, h_dim, n_layers=1, dropout=0.1):
        """
        :param vocab_dim: Number of vocabulary dimensions.
        :param h_dim: Number of hidden state dimensions.
        :param embedding_dim: Number of embedding dimensions.
        :param n_layers: Number of layer in the model.
        :param dropout: Level of dropout to apply between layers. Zero
        disables.
        """
        super().__init__()
        assert h_dim > 0 and n_layers > 0

        # the first phase:
        self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix)

        # self.linear1 = nn.Linear(embedding_dim, h_dim)

        # self.pe = PositionalEncoder(embedding_dim)

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=h_dim, num_layers=n_layers, dropout=dropout)

    def forward(self, x, **kw):

        embedded = self.embedding(x)
        # embedded = self.linear1(embedded)
        # embedded = self.pe(embedded)
        h, ht = self.lstm(embedded)

        return h, ht


class PeStanceLSTM(nn.Module):
    def __init__(self, weights_matrix, h_dim, n_layers=1, dropout=0.1):
        """
        :param vocab_dim: Number of vocabulary dimensions.
        :param h_dim: Number of hidden state dimensions.
        :param embedding_dim: Number of embedding dimensions.
        :param n_layers: Number of layer in the model.
        :param dropout: Level of dropout to apply between layers. Zero
        disables.
        """
        super().__init__()
        assert h_dim > 0 and n_layers > 0

        # the second phase:
        self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix)

        # self.linear1 = nn.Linear(embedding_dim, h_dim)

        self.ff = FeedForward(h_dim, h_dim*2)

        self.attn = torch.nn.MultiheadAttention(h_dim, 1)

        self.lstm = nn.LSTM(input_size=embedding_dim + h_dim*n_layers, hidden_size=h_dim, num_layers=n_layers, dropout=dropout)

        # self.linear = nn.Linear(h_dim, 3)

    def forward(self, x, h_prev, sub_h, **kw):
        S, B = x.shape
        embedded = self.embedding(x)  # embedded shape: (S, B, E)
        # print(embedded.shape)
        # embedded = self.linear1(embedded)

        # embedded = self.ff(embedded)
        q = h_prev[0]# (L,B, H)
        # print(q.shape)
        # q = self.pe(q)
        kv = sub_h# (S, B, H)
        kv = self.ff(kv)
        # print(kv.shape)
        a, _ = self.attn(q, kv, kv)  # (L, B , H)

        a = a.reshape((1, B, -1)).expand(S, -1, -1)
        rnn_input = torch.cat((embedded, a), dim=2)  # (S, B, E + L*H)

        # h:  (S, B, H)
        # ht: (L, B, H)
        h, ht = self.lstm(rnn_input, h_prev)

        # Project H back to the output size to get prediction for the stance
        # out = self.linear(h)

        # Out shapes: (S, B, V) and (L, B, H)
        return h, ht


class PeTwoPhaseLSTM(nn.Module):

    """
    Represents a two-phase LSTM model.
    """

    def __init__(self, weights_matrix, h_dim, n_layers=1, dropout=0.1):
        super().__init__()
        self.sub = PeSubjectivityLSTM(weights_matrix, h_dim, n_layers, dropout)
        self.stance = PeStanceLSTM(weights_matrix, h_dim, n_layers, dropout)
        self.pe = PositionalEncoder(h_dim)
        self.linear1 = nn.Linear(h_dim, 3)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.linear2 = nn.Linear(h_dim, 3)


    def forward(self, x_src, **kw):
        S, B = x_src.shape
        # context is (L, B, H)
        h, context = self.sub(x_src, **kw)
        stance_outputs = []
        stance_input = x_src[[0], :]
        h = self.pe(h)
        for t in range(1, S):
            # Feed the stance sequences of length 1 & save new context
            # stance_output is (1, B, V)
            stance_output, context = self.stance(stance_input, context, h, **kw)
            stance_outputs.append(stance_output)
            stance_input = x_src[[t], :]

        stance_out = torch.cat(stance_outputs, dim=0)
        stance_out = torch.mean(stance_out, 0)
        out = self.linear1(stance_out)
        y_stance = self.log_softmax(out)
        h = torch.mean(h, 0)
        sub_output = self.linear2(h)
        y_sub = self.log_softmax(sub_output)
        # Output shape: (B, 3)
        return y_stance, y_sub

class BaseLineLSTM(nn.Module):
    def __init__(self, vocab_dim, embedding_dim, h_dim, n_layers=1, dropout=0):
        """
        :param vocab_dim: Number of vocabulary dimensions.
        :param h_dim: Number of hidden state dimensions.
        :param embedding_dim: Number of embedding dimensions.
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

        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, **kw):

        embedded = self.embedding(x)

        h, ht = self.lstm(embedded)

        h = torch.mean(h, 0)

        out = self.linear(h)

        out = self.log_softmax(out)

        return out

class BaseLineRNN(nn.Module):
    def __init__(self, vocab_dim, embedding_dim, h_dim, n_layers=1, dropout=0.1):
        """
        :param vocab_dim: Number of vocabulary dimensions.
        :param h_dim: Number of hidden state dimensions.
        :param embedding_dim: Number of embedding dimensions.
        :param n_layers: Number of layer in the model.
        :param dropout: Level of dropout to apply between layers. Zero
        disables.
        """
        super().__init__()
        assert vocab_dim > 0 and h_dim > 0 and n_layers > 0

        # the first phase:
        self.embedding = nn.Embedding(vocab_dim, embedding_dim)

        self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=h_dim, num_layers=n_layers, dropout=dropout)

        self.linear = nn.Linear(h_dim, 3)

        self.log_softmax = nn.LogSoftmax(dim=1)


    def forward(self, x, **kw):

        embedded = self.embedding(x)

        h, ht = self.rnn(embedded)

        h = torch.mean(h, 0)

        out = self.linear(h)

        out = self.log_softmax(out)

        return out