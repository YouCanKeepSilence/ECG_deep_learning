import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, num_classes, count_of_features):
        super(MLP, self).__init__()

        self.num_classes = num_classes
        self.fc0 = nn.Linear(count_of_features, 784)  # полносвязный слой
        self.fc1 = nn.Linear(784, 100)  # полносвязный слой
        self.fc2 = nn.Linear(100, 50)  # полносвязный слой
        self.fc3 = nn.Linear(50, num_classes)  # полносвязный слой

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # in reason of CrossEntropy here should be softmax activate function instead of ReLU
        return x


class CNN(nn.Module):
    def __init__(self, num_classes, count_of_features):
        super(CNN, self).__init__()
        self.num_classes = num_classes
        conv_1_output_size = int(count_of_features / 2)
        conv_2_output_size = int(conv_1_output_size / 2)
        conv_3_output_size = int(conv_2_output_size / 2)

        self.conv1 = nn.Conv2d(count_of_features, conv_1_output_size, 3)
        self.conv2 = nn.Conv2d(conv_1_output_size, conv_2_output_size, 3)
        self.conv3 = nn.Conv2d(conv_2_output_size, conv_3_output_size, 3)
        self.classifier = nn.Linear(conv_3_output_size, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x