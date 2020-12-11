import torch
import torch.nn as nn

import utils.config as config
import model.text_model as text_model
import model.vision_model as vision_model


class Net(nn.Module):
    """
        embeddings: pre-trained embedding weights or num of embeddings.
    """
    def __init__(self, embeddings=None):
        super(Net, self).__init__()
        question_features = 1024
        vision_features = config.output_features
        visual_glimpses = 2

        self.text = text_model.TextProcessor(
            embeddings=embeddings,
            embedding_features=config.text_embed_size,
            gru_features=question_features,
            drop=0.5
        )
        self.vision_att = vision_model.VisualAtt(
            v_features=vision_features,
            q_features=question_features,
            mid_features=512,
            glimpses=visual_glimpses,
            drop=0.5,
        )
        self.classifier = Classifier(
            in_features=(visual_glimpses * vision_features + question_features),
            mid_features=1024,
            out_features=config.max_answers,
            drop=0.5,
        )

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, v, b, q, q_len):
        q = self.text(q, list(q_len.data))

        v = v / (v.norm(p=2, dim=1, keepdim=True) + 1e-12).expand_as(v)

        a = self.vision_att(v, q)
        v, v_att = vision_model.apply_attention(v, a)
        combined = torch.cat([v, q], dim=-1)
        answer = self.classifier(combined)

        return answer, v_att


class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU()
        self.lin11 = nn.Linear(in_features, mid_features)
        self.lin2 = nn.Linear(mid_features, out_features)

        self.kernel = nn.Parameter(torch.Tensor(mid_features, out_features))
        self.kernel.data.uniform_(-1 / mid_features, 1 / mid_features)

    def forward(self, x):
        vq = self.drop(self.relu(self.lin11(self.drop(x))))
        # vq = self.lin2(vq)

        if config.use_cos:
            k_norm = l2_norm(self.kernel, dim=0)
            vq = l2_norm(vq, dim=-1)
        else:
            k_norm = self.kernel
        x = torch.mm(vq, k_norm)
        return x


def l2_norm(input, dim=-1):
    norm = torch.norm(input, dim=dim, keepdim=True)
    output = torch.div(input, norm)
    return output
