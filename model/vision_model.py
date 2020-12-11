import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualAtt(nn.Module):
    def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
        super(VisualAtt, self).__init__()
        self.v_conv = nn.Conv1d(v_features, mid_features, 1, bias=False)  
        self.q_lin = nn.Linear(q_features, mid_features)
        self.x_conv = nn.Conv1d(mid_features, glimpses, 1)

        self.drop = nn.Dropout(drop)
        self.fusion = Fusion()

    def forward(self, v, q):
        v = self.v_conv(self.drop(v)) # [b, m, o]
        q = self.q_lin(self.drop(q)).unsqueeze(-1) # [b, m, 1]
        x = self.fusion(v, q)
        x = self.x_conv(self.drop(x)) # [b, g, o]
        return x


def apply_attention(input, attention):
    """ Apply any number of attention maps over the input. """
    b = input.size(0)
    attention = F.softmax(attention, dim=-1).unsqueeze(2) # [b, g, 1, o]
    input = input.unsqueeze(1) # [b, 1, v, o]
    weighted = attention * input # [b, g, v, o]
    weighted_mean = weighted.sum(dim=-1) # [b, g, v]
    return weighted_mean.view(b, -1), attention
   

class Fusion(nn.Module):
    """ Crazy multi-modal fusion: negative squared difference minus relu'd sum
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        # found through grad student descent ;)
        return - (x - y)**2 + F.relu(x + y)
