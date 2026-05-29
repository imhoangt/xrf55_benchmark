"""ResNet18 1D — adapted from XRF55 official repo for HAR-CSI baseline.

Input:  (B, inchannel, T)
Output: (B, num_classes)
"""
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, group=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, groups=group)


def conv1x1(in_planes, out_planes, stride=1, group=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False, groups=group)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, group=1, downsample=None):
        super().__init__()
        self.conv1      = conv3x3(inplanes, planes, stride, group)
        self.bn1        = nn.BatchNorm1d(planes)
        self.relu       = nn.ReLU(inplace=True)
        self.conv2      = conv3x3(planes, planes, group=group)
        self.bn2        = nn.BatchNorm1d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet(nn.Module):
    def __init__(self, block, layers, inchannel=270, activity_num=11):
        super().__init__()
        self.inplanes = 128
        self.conv1    = nn.Conv1d(inchannel, 128, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv2    = nn.Conv1d(128,       128, kernel_size=7, stride=1, padding=3, bias=False)
        self.conv3    = nn.Conv1d(128,       128, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1      = nn.BatchNorm1d(128)
        self.bn2      = nn.BatchNorm1d(128)
        self.bn3      = nn.BatchNorm1d(128)
        self.relu     = nn.ReLU(inplace=True)
        self.maxpool  = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1   = self._make_layer(block, 64,  layers[0])
        self.layer2   = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3   = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4   = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool  = nn.AdaptiveAvgPool1d(1)
        self.fc       = nn.Linear(512 * block.expansion, activity_num)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)

    def _make_layer(self, block, planes, blocks, stride=1, group=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, group, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, group=group))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return self.fc(x.flatten(1))


def resnet18(inchannel=270, num_classes=11):
    return ResNet(BasicBlock, [2, 2, 2, 2], inchannel=inchannel, activity_num=num_classes)
