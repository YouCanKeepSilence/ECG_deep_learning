import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, num_classes=9, count_of_features=12 * 2500 + 2):
        super(MLP, self).__init__()

        self.num_classes = num_classes
        self.input_bn = nn.BatchNorm1d(count_of_features)
        self.fc0 = nn.Linear(count_of_features, 784)
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, non_ecg, ecg):
        x = ecg
        x = x.view(x.size(0), -1)
        x = torch.cat((non_ecg, x), dim=1)
        x = self.input_bn(x)
        x = F.relu(self.fc0(x))
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # we don't need activation here in reason of CrossEntropyLoss usage. It includes LogSoftmax inside
        return x


class CNNFromArticle(nn.Module):
    def __init__(self, num_classes=9, number_of_channels=12):
        super(CNNFromArticle, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm1d(number_of_channels),
            nn.Conv1d(number_of_channels, 24, 5, stride=1, padding=0),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.MaxPool1d(5),
            nn.Conv1d(24, 24, 5, stride=1, padding=0),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.MaxPool1d(5),
            nn.Conv1d(24, 36, 5, stride=1, padding=0),
            nn.BatchNorm1d(36),
            nn.ReLU(),
            nn.MaxPool1d(5),
            nn.Conv1d(36, 36, 5, stride=1, padding=0),
            nn.BatchNorm1d(36),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.BatchNorm1d(542),
            nn.Linear(542, 100),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(100, num_classes)
        )

    def forward(self, non_ecg, ecg):
        x = self.conv(ecg)
        x = x.view(x.size(0), -1)
        x = torch.cat((non_ecg, x), dim=1)
        # print(x.size())
        self.fc(x)
        # we don't need activation here in reason of CrossEntropyLoss usage. It includes LogSoftmax inside
        return x


class CNN(nn.Module):
    def __init__(self, num_classes=9, number_of_channels=12, pooling='max'):
        super(CNN, self).__init__()
        if pooling == 'avg':
            self.pooling = nn.AvgPool1d(kernel_size=3, stride=3)
        else:
            self.pooling = nn.MaxPool1d(kernel_size=3, stride=3)

        self.num_classes = num_classes
        self.dropout05 = nn.Dropout(0.5)
        self.dropout025 = nn.Dropout(0.25)
        self.input_bn = nn.BatchNorm1d(number_of_channels)
        self.fc_input_bn = nn.BatchNorm1d(8736 + 2)
        self.conv_1_layer = nn.Sequential(
            nn.Conv1d(12, 24, kernel_size=3),
            nn.BatchNorm1d(24),
            nn.ReLU(inplace=True),
            self.pooling
        )
        self.conv_2_layer = nn.Sequential(
            nn.Conv1d(24, 48, kernel_size=3),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
            self.pooling
        )
        self.conv_3_layer = nn.Sequential(
            nn.Conv1d(48, 96, kernel_size=3),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
            self.pooling
        )
        self.fc_1_layer = nn.Sequential(
            nn.Linear(8736 + 2, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
        )
        self.fc_2_layer = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )
        self.fc_3_layer = nn.Sequential(
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )
        self.fc4 = nn.Linear(256, num_classes)

    def forward(self, non_ecg, ecg):
        x = ecg
        x = self.input_bn(x)
        x = self.conv_1_layer(x)
        x = self.conv_2_layer(x)
        x = self.dropout025(x)
        x = self.conv_3_layer(x)
        x = x.view(x.size(0), -1)
        x = torch.cat((non_ecg, x), dim=1)
        x = self.fc_input_bn(x)
        x = self.dropout025(x)
        x = self.fc_1_layer(x)
        x = self.dropout05(x)
        x = self.fc_2_layer(x)
        x = self.dropout025(x)
        x = self.fc_3_layer(x)
        x = self.fc4(x)
        # we don't need activation here in reason of CrossEntropyLoss usage. It includes LogSoftmax inside
        return x


class VGGLikeCNN(nn.Module):
    def __init__(self, num_classes=9, number_of_channels=12, pooling='avg'):
        super().__init__()
        if pooling == 'avg':
            self.pooling = nn.AvgPool1d(kernel_size=3, stride=3)
        else:
            self.pooling = nn.MaxPool1d(kernel_size=3, stride=3)

        self.num_classes = num_classes
        self.dropout05 = nn.Dropout(0.5)
        self.dropout025 = nn.Dropout(0.25)
        self.input_bn = nn.BatchNorm1d(number_of_channels)
        self.fc_input_bn = nn.BatchNorm1d(8640 + 2)
        self.conv_1_layer = nn.Sequential(
            nn.Conv1d(12, 24, kernel_size=3),
            nn.BatchNorm1d(24),
            nn.ReLU(inplace=True),
            nn.Conv1d(24, 24, kernel_size=3),
            nn.BatchNorm1d(24),
            nn.ReLU(inplace=True),
            self.pooling
        )
        self.conv_2_layer = nn.Sequential(
            nn.Conv1d(24, 48, kernel_size=3),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
            nn.Conv1d(48, 48, kernel_size=3),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
            self.pooling
        )
        self.conv_3_layer = nn.Sequential(
            nn.Conv1d(48, 96, kernel_size=3),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
            nn.Conv1d(96, 96, kernel_size=3),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
            self.pooling
        )
        self.fc_1_layer = nn.Sequential(
            nn.Linear(8640 + 2, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
        )
        self.fc_2_layer = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
        )
        self.fc_3_layer = nn.Linear(2048, num_classes)

    def forward(self, non_ecg, ecg):
        x = ecg
        x = self.input_bn(x)
        x = self.conv_1_layer(x)
        x = self.conv_2_layer(x)
        x = self.conv_3_layer(x)
        x = x.view(x.size(0), -1)
        x = torch.cat((non_ecg, x), dim=1)
        x = self.fc_input_bn(x)
        x = self.fc_1_layer(x)
        x = self.dropout05(x)
        x = self.fc_2_layer(x)
        x = self.dropout05(x)
        x = self.fc_3_layer(x)
        # we don't need activation here in reason of CrossEntropyLoss usage. It includes LogSoftmax inside
        return x


vgg_cfgs = {
    '11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '11a': [64, 'A', 128, 'A', 256, 256, 'A', 512, 512, 'A', 512, 512, 'A'],
    '13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '13a': [64, 64, 'A', 128, 128, 'A', 256, 256, 'A', 512, 512, 'A', 512, 512, 'A'],
    '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    '16a': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 'A', 512, 512, 512, 'A', 512, 512, 512, 'A'],
    '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    '19a': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 256, 'A', 512, 512, 512, 512, 'A', 512, 512, 512, 512, 'A']
}


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 12
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
        elif v == 'A':
            layers += [nn.AvgPool1d(kernel_size=2, stride=2)]
        else:
            conv1d = nn.Conv1d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv1d, nn.BatchNorm1d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv1d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGGGeneric(nn.Module):
    def __init__(self, features, number_of_channels=12, num_classes=9, init_weights=True):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(number_of_channels)
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool1d(7)
        self.classifier_bn = nn.BatchNorm1d(512 * 7 + 2)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 + 2, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, non_ecg, ecg):
        x = self.input_bn(ecg)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = torch.cat((non_ecg, x), dim=1)
        x = self.classifier_bn(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def get_vgg(version='11', batch_norm=False, **kwargs):
    return VGGGeneric(make_layers(vgg_cfgs[version], batch_norm=batch_norm), **kwargs)
