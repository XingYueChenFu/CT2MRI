import torch
import torch.nn as nn
from collections import OrderedDict

def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, shortcut_type='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.shortcut_type = shortcut_type

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            if self.shortcut_type == 'B':
                residual = self.downsample(x)
            elif self.shortcut_type == 'A':
                residual = nn.functional.avg_pool3d(x, kernel_size=self.stride, stride=self.stride)
                residual = torch.cat([residual] * (out.shape[1] // x.shape[1]), dim=1)
                if residual.shape[1] != out.shape[1]:
                    residual = torch.cat([residual] * 2, dim=1)[:, :out.shape[1], ...]

        out += residual
        out = self.relu(out)

        return out

class ResNet18_3D(nn.Module):
    def __init__(self, shortcut_type='A', no_cuda=False):
        super(ResNet18_3D, self).__init__()
        self.inplanes = 64
        self.shortcut_type = shortcut_type
        
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
            if self.shortcut_type == 'B':
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * BasicBlock.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm3d(planes * BasicBlock.expansion),
                )
        
        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample, self.shortcut_type))
        self.inplanes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes, shortcut_type=self.shortcut_type))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x4

def resnet18(shortcut_type='A', no_cuda=False):
    """Constructs a ResNet-18 model."""
    model = ResNet18_3D(shortcut_type, no_cuda)
    return model