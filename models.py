import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, num_classes, count_of_features):
        super(MLP, self).__init__()

        self.num_classes = num_classes
        self.fc0 = nn.Linear(count_of_features, 784)  # полносвязный слой
        self.fc1 = nn.Linear(784, 512)  # полносвязный слой
        self.fc2 = nn.Linear(512, 256)  # полносвязный слой
        self.fc3 = nn.Linear(256, num_classes)  # полносвязный слой
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc0(x))
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # x = self.fc2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # in reason of CrossEntropy here should be softmax activate function instead of ReLU
        return F.log_softmax(x, dim=1)


class CNN(nn.Module):
    def __init__(self, num_classes, number_of_channels=12):
        super(CNN, self).__init__()
        self.num_classes = num_classes
        self.input_bn = nn.BatchNorm1d(number_of_channels)
        self.conv1 = nn.Conv1d(12, 24, kernel_size=3)
        self.conv2 = nn.Conv1d(24, 48, kernel_size=3)
        self.conv3 = nn.Conv1d(48, 96, kernel_size=3)
        self.pooling = nn.AvgPool1d(kernel_size=3, stride=3)
        self.dropout = nn.Dropout(0.5)
        self.fc_bn = nn.BatchNorm1d(8736 + 2)
        self.fc1 = nn.Linear(8736 + 2, 784)
        self.fc2 = nn.Linear(784, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, non_ecg, ecg):
        x = ecg
        x = self.input_bn(x)
        x = F.relu(self.conv1(x))
        x = self.pooling(x)
        x = F.relu(self.conv2(x))
        x = self.pooling(x)
        x = F.relu(self.conv3(x))
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = torch.cat((non_ecg, x), dim=1)
        x = self.fc_bn(x)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
