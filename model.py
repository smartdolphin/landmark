import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Conv2d, AdaptiveAvgPool2d, Linear
from torch.nn.parameter import Parameter
from efficientnet_pytorch import EfficientNet


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features):
        super(ArcMarginProduct, self).__init__()
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class EfficientNetEncoderHead(nn.Module):
    def __init__(self, depth, num_classes, feat_dim=512):
        super(EfficientNetEncoderHead, self).__init__()
        self.depth = depth
        self.base = EfficientNet.from_pretrained(f'efficientnet-b{self.depth}')
        self.gem = GeM()
        self.output_filter = self.base._fc.in_features
        self.neck = nn.Sequential(
            nn.Linear(self.output_filter, feat_dim),
            nn.BatchNorm1d(feat_dim),
            torch.nn.PReLU(),
        )
        self.head = ArcMarginProduct(feat_dim, num_classes)

    def forward(self, x, label=None):
        x = self.base.extract_features(x)
        x = self.gem(x).squeeze()
        x = self.neck(x)
        logits = self.head(x)
        return logits

