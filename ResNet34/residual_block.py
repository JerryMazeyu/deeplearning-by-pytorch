import torch as t
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut


    def forward(self, x):
        x_1 = self.left(x)
        if self.right == None:
            x_2 = x
        else:
            x_2 = self.right(x)
        x = x_1 + x_2
        return F.relu(x)