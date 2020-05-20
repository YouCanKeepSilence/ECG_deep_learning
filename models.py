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


class CNN(nn.Module):
    def __init__(self, num_classes=9, number_of_channels=12, pooling='avg'):
        super(CNN, self).__init__()
        if pooling == 'avg':
            self.pooling = nn.AvgPool1d(kernel_size=3, stride=3)
        else:
            self.pooling = nn.MaxPool1d(kernel_size=3, stride=3)

        self.dropout05 = nn.Dropout(0.5)
        self.dropout025 = nn.Dropout(0.25)
        self.num_classes = num_classes
        self.conv_1_layer = nn.Sequential(
            nn.Conv1d(12, 24, kernel_size=3),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            self.pooling
        )
        self.conv_2_layer = nn.Sequential(
            nn.Conv1d(24, 48, kernel_size=3),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            self.pooling
        )
        self.conv_3_layer = nn.Sequential(
            nn.Conv1d(48, 96, kernel_size=3),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            self.pooling
        )
        self.input_bn = nn.BatchNorm1d(number_of_channels)

        self.fc_input_bn = nn.BatchNorm1d(8736 + 2)
        self.fc1 = nn.Linear(8736 + 2, 2048)
        self.fc1_bn = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc2_bn = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 256)
        self.fc3_bn = nn.BatchNorm1d(256)
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
        x = self.dropout05(x)
        x = F.relu(self.fc1(x))
        x = self.fc1_bn(x)
        x = F.relu(self.fc2(x))
        x = self.dropout05(x)
        x = self.fc2_bn(x)
        x = self.dropout025(x)
        x = F.relu(self.fc3(x))
        x = self.fc3_bn(x)
        x = self.fc4(x)
        # we don't need activation here in reason of CrossEntropyLoss usage. It includes LogSoftmax inside
        return x
