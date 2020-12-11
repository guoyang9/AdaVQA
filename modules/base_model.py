import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

import utils.config as config
from modules.fc import FCNet
from modules.attention import Attention, NewAttention
from modules.language_model import WordEmbedding, QuestionEmbedding


class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, fusion, num_hid, num_class):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.fusion = fusion
        # self.classifier = weight_norm(nn.Linear(num_hid*2, num_class), dim=None)
        num_hid = num_hid * 2
        self.kernel = nn.Parameter(torch.Tensor(num_hid, num_class))
        self.kernel.data.uniform_(-1 / num_hid, 1 / num_hid)

    def forward(self, v, q):
        """
        Forward=

        v: [batch, num_objs, obj_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb, _ = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        fusion_repr = self.fusion(joint_repr)
        # logits = self.classifier(logits)

        if config.use_cos:
            k_norm = l2_norm(self.kernel, dim=0)
            fusion_repr = l2_norm(fusion_repr, dim=-1)
        else:
            k_norm = self.kernel
        logits = torch.mm(fusion_repr, k_norm)

        return logits, joint_repr


def l2_norm(input, dim=-1):
    norm = torch.norm(input, dim=dim, keepdim=True)
    output = torch.div(input, norm)
    return output


def build_baseline(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    fusion = FCNet([num_hid, num_hid*2], dropout=0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net,
                     fusion, num_hid, dataset.num_ans_candidates)


def build_baseline_newatt(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    fusion = FCNet([num_hid, num_hid*2], dropout=0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net,
                     fusion, num_hid, dataset.num_ans_candidates)
