import math
import torch
from torchvision import models
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn as nn
from torch.autograd import Variable
from .heads import Corr_Up, MultiRPN, DepthwiseRPN
from .backbones import AlexNet, Vgg, ResNet22, Incep22, ResNeXt22, ResNet22W, resnet50, resnet34, resnet18
from neck import AdjustLayer, AdjustAllLayer
from .utils import load_pretrain

__all__ = ['SiamFC_', 'SiamFC', 'SiamVGG', 'SiamFCRes22', 'SiamFCIncep22', 'SiamFCNext22', 'SiamFCRes22W',
           'SiamRPN', 'SiamRPNVGG', 'SiamRPNRes22', 'SiamRPNIncep22', 'SiamRPNResNeXt22', 'SiamRPNPP']

class SiamFC_(nn.Module):
    def __init__(self):
        super(SiamFC_, self).__init__()
        self.features = None
        # self.head = None

    def head(self, z, x):
        n, c, h, w = x.size()
        x = x.view(1, n * c, h, w)
        out = F.conv2d(x, z, groups=n)
        out = out.view(n, 1, out.size(-2), out.size(-1))
        return out

    def feature_extractor(self, x):
        return self.features(x)

    def connector(self, template_feature, search_feature):
        pred_score = self.head(template_feature, search_feature)
        return pred_score

    def branch(self, allin):
        allout = self.feature_extractor(allin)
        return allout

    def forward(self, template, search):
        zf = self.feature_extractor(template)
        xf = self.feature_extractor(search)
        score = self.connector(zf, xf)
        return score


class SiamFC(SiamFC_):

    def __init__(self):
        super(SiamFC, self).__init__()
        self.features = AlexNet()
        self._initialize_weights()

    def forward(self, z, x):
        zf = self.features(z)
        xf = self.features(x)
        score = self.head(zf, xf)
        return score

    def head(self, z, x):
        # fast cross correlation
        n, c, h, w = x.size()
        x = x.view(1, n * c, h, w)
        out = F.conv2d(x, z, groups=n)
        out = out.view(n, 1, out.size(-2), out.size(-1))

        # adjust the scale of responses
        out = 0.001 * out + 0.0

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data, mode='fan_out',
                                     nonlinearity='relu')
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

                
class SiamVGG(nn.Module):

    def __init__(self):
        super(SiamVGG, self).__init__()
        self.features = Vgg()
        self.bn_adjust = nn.BatchNorm2d(1)
        self._initialize_weights()

        # init weight with pretrained model
        mod = models.vgg16(pretrained=True)
        for i in xrange(len(self.features.state_dict().items()) - 2):
            self.features.state_dict().items()[i][1].data[:] = mod.state_dict().items()[i][1].data[:]

    def forward(self, z, x):
        zf = self.features(z)
        xf = self.features(x)
        score = self.head(zf, xf)

        return score

    def head(self, z, x):
        out = []
        for i in range(x.size(0)):
            out.append(F.conv2d(x[i, :, :, :].unsqueeze(0), z[i, :, :, :].unsqueeze(0)))

        return self.bn_adjust(torch.cat(out, dim=0))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
