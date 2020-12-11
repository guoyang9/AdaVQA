import torch
import torch.nn as nn

import utils.config as config
import model.counting as counting
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
        textual_glimpses = 1
        objects = 10

        self.text = text_model.TextProcessor(
            embeddings=embeddings,
            embedding_features=config.text_embed_size,
            gru_features=question_features,
            drop=0.5
        )
        # self.text_att = text_model.TextualAtt(
        #     in_features=question_features,
        #     mid_features=512,
        #     glimpses=textual_glimpses,
        #     drop=0.5,
        # )
        self.vision_att = vision_model.VisualAtt(
            v_features=vision_features,
            q_features=question_features,
            # q_features=textual_glimpses * question_features,
            mid_features=512,
            glimpses=visual_glimpses,
            drop=0.5,
        )
        self.classifier = Classifier(
            in_features=(visual_glimpses * vision_features, question_features),
            # in_features=(visual_glimpses * vision_features, textual_glimpses * question_features),
            mid_features=1024,
            out_features=config.max_answers,
            count_features=objects + 1,
            drop=0.5,
        )
        self.counter = counting.Counter(objects)

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, v, b, q, q_len):
        q = self.text(q, list(q_len.data))
        # q, q_att = self.text_att(q)

        v = v / (v.norm(p=2, dim=1, keepdim=True) + 1e-12).expand_as(v)

        a = self.vision_att(v, q)
        v, v_att = vision_model.apply_attention(v, a)

        # this is where the counting component is used
        # pick out the first attention map
        a1 = a[:, 0, :].contiguous().view(a.size(0), -1)
        # give it and the bounding boxes to the component
        count = self.counter(b, a1) if config.image_feature == 'rcnn' else None
        # count = None
        answer, _ = self.classifier(v, q, count)

        return answer, v_att


class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, count_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU()
        self.fusion = vision_model.Fusion()
        self.lin11 = nn.Linear(in_features[0], mid_features)
        self.lin12 = nn.Linear(in_features[1], mid_features)
        self.lin2 = nn.Linear(mid_features, out_features)
        self.lin_c = nn.Linear(count_features, mid_features)
        self.bn = nn.BatchNorm1d(mid_features)
        self.bn2 = nn.BatchNorm1d(mid_features)

        self.kernel = nn.Parameter(torch.Tensor(mid_features, out_features))
        self.kernel.data.uniform_(-1 / mid_features, 1 / mid_features)

    def forward(self, x, y, c):
        vq = self.fusion(self.lin11(self.drop(x)), self.lin12(self.drop(y)))
        if torch.is_tensor(c):
            vq = vq + self.bn2(self.relu(self.lin_c(c)))
        fusion_repr = self.drop(self.bn(vq))
        # x = self.lin2(fusion_repr)

        if config.use_cos:
            k_norm = l2_norm(self.kernel, dim=0)
            fusion_repr = l2_norm(fusion_repr, dim=-1)
        else:
            k_norm = self.kernel
        x = torch.mm(fusion_repr, k_norm)
        return x, vq


def l2_norm(input, dim=-1):
    norm = torch.norm(input, dim=dim, keepdim=True)
    output = torch.div(input, norm)
    return output
