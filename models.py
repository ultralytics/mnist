# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch.nn as nn
import torch.nn.functional as F


class SANDD(nn.Module):
    """Implements a convolutional neural network for signal anomaly detection and diagnosis (SANDD)."""

    def __init__(self, n_out=2):
        """Initializes the SANDD model with optional output layer size."""
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 30), stride=(1, 2), padding=(0, 15), bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=(1, 2), stride=1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 30), stride=(1, 2), padding=(0, 15), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=(1, 2), stride=1),
        )
        self.layer3 = nn.Sequential(nn.Conv2d(64, n_out, kernel_size=(1, 100), stride=(1, 1), padding=(0, 0)))

    def forward(self, x):  # x.shape = [bs, 400]
        """Defines the forward pass of the neural network, transforming input tensor x of shape [bs, 400] to output
        shape [bs, 1].
        """
        x = x.view((-1, 1, 400))  # [bs, 1, 400]
        x = x.unsqueeze(1)  # [bs, 1, 1, 400]
        x = self.layer1(x)  # [bs, 32, 1, 200]
        x = self.layer2(x)  # [bs, 64, 1, 100]
        x = self.layer3(x)  # [bs, 1, 1, 1]
        return x.reshape(x.size(0), -1)  # [bs, 1]


#       121  2.6941e-05    0.021642      11.923     0.14201  # var 1
class WAVE2(nn.Module):
    """A CNN model for processing 2D input data with batch normalization, activation, and pooling layers."""

    def __init__(self, n_out=2):
        """Initializes the WAVE2 model with convolutional, batch normalization, activation, and pooling layers."""
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(2, 30), stride=(1, 2), padding=(1, 15), bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=(1, 2), stride=1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(2, 30), stride=(1, 2), padding=(0, 15), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=(1, 2), stride=1),
        )
        self.layer3 = nn.Sequential(nn.Conv2d(64, n_out, kernel_size=(2, 64), stride=(1, 1), padding=(0, 0)))

    def forward(self, x):  # x.shape = [bs, 512]
        """Forward pass for processing input tensor x through sequential layers, reshaping as needed; x.shape = [bs,
        512].
        """
        x = x.view((-1, 2, 256))  # [bs, 2, 256]
        x = x.unsqueeze(1)  # [bs, 1, 2, 256]
        x = self.layer1(x)  # [bs, 32, 1, 128]
        x = self.layer2(x)  # [bs, 64, 1, 64]
        x = self.layer3(x)
        return x.reshape(x.size(0), -1)  # [bs, 64*64]


# Epoch 25: 98.60% test accuracy, 0.0555 test loss (normalize after relu)
# Epoch 11: 98.48% test accuracy, 0.0551 test loss (normalize after both)
class MLP(nn.Module):
    """A simple MLP model with two fully connected layers for classification tasks."""

    def __init__(self):
        """Initialize MLP model with two fully connected layers."""
        super().__init__()
        self.fc1 = nn.Linear(784, 500, bias=True)
        self.fc2 = nn.Linear(500, 10, bias=True)

    def forward(self, x):
        """Pass input through two fully connected layers with ReLU activation in between, returning the output."""
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = F.relu(x)
        # x, _, _ = normalize(x, axis=1)
        x = self.fc2(x)
        return x


# 178  9.2745e-05    0.024801        99.2 default no augmentation
class ConvNeta(nn.Module):
    """A convolutional neural network model with dropout and fully connected layers for image classification tasks."""

    def __init__(self):
        """Initializes the ConvNeta neural network architecture with convolutional, dropout, and fully connected
        layers.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        """Defines the forward pass of the ConvNet model applying convolutional, pooling, dropout, and fully connected
        layers.
        """
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


# https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/02-intermediate
# 8    0.00023365    0.025934       99.14  default no augmentation
# 124      14.438    0.012876       99.55  LeakyReLU in place of ReLU
# 190  0.00059581    0.013831       99.58  default
class ConvNetb(nn.Module):
    """Implements a CNN with two convolutional layers and a fully connected layer for classification tasks."""

    def __init__(self, num_classes=10):
        """Initialize ConvNetb layers with given number of output classes."""
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):  # x.size() = [512, 1, 28, 28]
        """Performs forward pass through two convolutional layers and a fully connected layer, transforming input x of
        shape [512, 1, 28, 28].
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        # x, _, _ = normalize(x,1)
        x = self.fc(x)
        # x = F.sigmoid(x)
        # x = F.log_softmax(x, dim=1)  # ONLY for use with nn.NLLLoss
        return x
