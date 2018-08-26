import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *

# import torchvision
# from torchvision import datasets, transforms

torch.set_printoptions(linewidth=320, precision=8)
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5


# Epoch 25: 98.60% test accuracy, 0.0555 test loss (normalize after relu)
# Epoch 11: 98.48% test accuracy, 0.0551 test loss (normalize after both)
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 500, bias=True)
        self.fc2 = nn.Linear(500, 10, bias=True)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = F.relu(x)
        # x, _, _ = normalize(x, axis=1)
        x = self.fc2(x)
        return x


# 178  9.2745e-05    0.024801        99.2 default no augmentation
class ConvNeta(nn.Module):
    def __init__(self):
        super(ConvNeta, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
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
    def __init__(self, num_classes=10):
        super(ConvNetb, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):  # x.size() = [512, 1, 28, 28]
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        # x, _, _ = normalize(x,1)
        x = self.fc(x)
        # x = F.sigmoid(x)
        # x = F.log_softmax(x, dim=1)  # ONLY for use with nn.NLLLoss
        return x


def main(model):
    lr = .001
    epochs = 10
    printerval = 1
    patience = 200
    batch_size = 1000
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    print('Running on %s\n%s' % (device.type, torch.cuda.get_device_properties(0) if cuda else ''))

    np.random.seed(0)
    torch.manual_seed(0)
    if cuda:
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    # MNIST Dataset
    # tforms = transforms.Compose([torchvision.transforms.RandomAffine(
    #    degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10)),
    #    transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # tformstest = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # train = datasets.MNIST(root='./data', train=True, transform=tforms, download=True)
    # test = datasets.MNIST(root='./data', train=False, transform=tformstest)
    # train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=False)
    # test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=10000, shuffle=False)

    # if not os.path.exists('data/MNISTtrain.mat'):
    #    scipy.io.savemat('data/MNISTtrain.mat',
    #                     {'x': train.train_data.unsqueeze(1).numpy(), 'y': train.train_labels.squeeze().numpy()})
    #    scipy.io.savemat('data/MNISTtest.mat',
    #                     {'x': test.test_data.unsqueeze(1).numpy(), 'y': test.test_labels.squeeze().numpy()})

    mat = scipy.io.loadmat('data/MNISTtrain.mat')
    train_loader2 = create_batches(x=torch.Tensor(mat['x']),
                                   y=torch.Tensor(mat['y']).squeeze().long(),
                                   batch_size=batch_size, shuffle=True)

    mat = scipy.io.loadmat('data/MNISTtest.mat')
    test_data = torch.Tensor(mat['x']), torch.Tensor(mat['y']).squeeze().long().to(device)
    # test_loader2 = create_batches(dataset=test_data, batch_size=10000)

    model = model.to(device)
    criteria1 = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    stopper = patienceStopper(epochs=epochs, patience=patience, printerval=printerval)

    def train(model):
        for i, (x, y) in enumerate(train_loader2):
            x, y = x.to(device), y.to(device)
            loss = criteria1(model(x), y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def test(model):
        x, y = test_data
        x, y = x.to(device), y.to(device)

        yhat = model(x)
        loss = criteria1(yhat, y)
        yhat_number = torch.argmax(yhat.data, 1)

        accuracy = []
        for i in range(10):
            j = y == i
            accuracy.append((yhat_number[j] == y[j]).float().mean() * 100.0)

        return loss, accuracy

    for epoch in range(epochs):
        train(model.train())
        loss, accuracy = test(model.eval())
        if stopper.step(loss, metrics=(*accuracy,), model=model):
            break



if __name__ == '__main__':
    # main(MLP())
    # main(ConvNeta())
    main(ConvNetb())
