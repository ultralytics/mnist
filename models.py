import torch.nn as nn
import torch.nn.functional as F

#       121  2.6941e-05    0.021642      11.923     0.14201  # var 1
class WAVE2(nn.Module):
    def __init__(self, n_out=2):
        super(WAVE2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(2, 30), stride=(1, 2), padding=(1, 15), bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=(1, 2), stride=1))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(2, 30), stride=(1, 2), padding=(0, 15), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=(1, 2), stride=1))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, n_out, kernel_size=(2, 64), stride=(1, 1), padding=(0, 0)))

    def forward(self, x):  # x.shape = [bs, 512]
        x = x.view((-1, 2, 256))  # [bs, 2, 256]
        x = x.unsqueeze(1)  # [bs, 1, 2, 256]
        x = self.layer1(x)  # [bs, 32, 1, 128]
        x = self.layer2(x)  # [bs, 64, 1, 64]
        x = self.layer3(x)
        return x.reshape(x.size(0), -1)  # [bs, 64*64]