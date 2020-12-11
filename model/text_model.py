import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import rnn

import utils.config as config
from model.pretrained_models import Bert
from model.pretrained_models import Glove


class TextProcessor(nn.Module):
    def __init__(self, embeddings, embedding_features, gru_features, drop=0.0):
        super(TextProcessor, self).__init__()
        self.drop = nn.Dropout(drop)
        self.tanh = nn.Tanh()
        self.gru = nn.GRU(input_size=embedding_features,
                           hidden_size=gru_features,
                           num_layers=1)

        self._init_gru(self.gru.weight_ih_l0)
        self._init_gru(self.gru.weight_hh_l0)
        self.gru.bias_ih_l0.data.zero_()
        self.gru.bias_hh_l0.data.zero_()

        if config.pretrained_model == 'bert':
            self.bert = Bert(config.bert_model, config.max_question_len)
        elif config.pretrained_model == 'glove':
            self.embedding = Glove(embeddings, fine_tune=True)
        else:
            self.embedding = nn.Embedding(embeddings, embedding_features)
            nn.init.xavier_uniform_(self.embedding.weight)

    def _init_gru(self, weight):
        for w in weight.chunk(3, 0):
            nn.init.xavier_uniform_(w)

    def forward(self, q, q_len):
        if config.pretrained_model == 'bert':
            tanhed = self.bert.forward(q[0], q[1])
        else:
            embedded = self.embedding(q)
            tanhed = self.tanh(self.drop(embedded))
            packed = rnn.pack_padded_sequence(tanhed, q_len, batch_first=True)
        o, h = self.gru(packed)
        o = rnn.pad_packed_sequence(o, total_length=config.max_question_len)
        return h.squeeze(0)
        # return o[0].transpose(0, 1)


class TextualAtt(nn.Module):
    """ question self-attention. """
    def __init__(self, in_features, mid_features, glimpses=1, drop=0.0):
        super(TextualAtt, self).__init__()
        self.conv1 = nn.Conv1d(in_features, mid_features, 1, bias=False)
        self.conv2 = nn.Conv1d(mid_features, glimpses, 1)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(drop)

    def forward(self, q):
        b, n = q.size()[:2]

        q_conv0 = q.transpose(1, 2) # [b, e, n]
        q_conv1 = self.relu(self.conv1(self.drop(q_conv0)))
        q_conv2 = self.conv2(self.drop(q_conv1)) # [b, 2, n]
        q_att = F.softmax(q_conv2, dim=-1)

        q = q.unsqueeze(2) # [b, n, 1, e]
        q_att = q_att.transpose(1, 2).unsqueeze(3) # [b, n, 2, 1]
        q = q * q_att
        q = q.contiguous().view(b, n, -1).sum(dim=1)
        return q, q_att
