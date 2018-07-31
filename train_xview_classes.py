import math
import random

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *

# import torchvision
# from torchvision import datasets, transforms

torch.set_printoptions(linewidth=320, precision=8)
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5


def xview_class_weights(indices):  # weights of each class in the training set, normalized to mu = 1
    weights = 1 / torch.FloatTensor(
        [74, 364, 713, 71, 2925, 20976.7, 6925, 1101, 3612, 12134, 5871, 3640, 860, 4062, 895, 149, 174, 17, 1624, 1846,
         125, 122, 124, 662, 1452, 697, 222, 190, 786, 200, 450, 295, 79, 205, 156, 181, 70, 64, 337, 1352, 336, 78,
         628, 841, 287, 83, 702, 1177, 31386.5, 195, 1081, 882, 1059, 4175, 123, 1700, 2317, 1579, 368, 85])
    weights /= weights.sum()
    return weights[indices]


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
    def __init__(self, num_classes=60):
        super(ConvNetb, self).__init__()
        n = 64  # initial convolution size
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, n, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(n, n * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(n * 2),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(n * 2, n * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(n * 4),
            nn.ReLU())
        # self.layer4 = nn.Sequential(
        #     nn.Conv2d(n * 4, n * 8, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(n * 8),
        #     nn.ReLU())
        # self.layer5 = nn.Sequential(
        #     nn.Conv2d(n * 8, n * 16, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(n * 16),
        #     nn.ReLU())
        # self.fc = nn.Linear(n * n * 8, num_classes)
        # self.fc = nn.Linear(8192, num_classes)  # chips32+14 3layer 32n
        # self.fc = nn.Linear(4096, num_classes)  # chips32+14 4layer 32n
        # self.fc = nn.Linear(12800, num_classes)  # chips40+16 3layer 32n
        # self.fc = nn.Linear(18432 * 2, num_classes)  # chips48+16 3layer 64n
        # self.fc = nn.Linear(4608, num_classes)  # chips48+16 3layer 32n
        self.fc = nn.Linear(65536, num_classes)  # 64+36, 3layer 64n
        # self.fc = nn.Linear(32768, num_classes)  # 64+36, 3layer 32n

    def forward(self, x):  # x.size() = [512, 1, 28, 28]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        # x = self.layer5(x)
        x = x.reshape(x.size(0), -1)
        # x, _, _ = normalize(x,1)
        x = self.fc(x)
        return x


# @profile
def main(model):
    lr = .0001
    epochs = 1000
    printerval = 1
    patience = 500
    batch_size = 500
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    print('Running on %s\n%s' % (device.type, torch.cuda.get_device_properties(0) if cuda else ''))

    rgb_mean = torch.FloatTensor([60.134, 49.697, 40.746]).view((1, 3, 1, 1)).to(device)
    rgb_std = torch.FloatTensor([29.99, 24.498, 22.046]).view((1, 3, 1, 1)).to(device)

    np.random.seed(0)
    torch.manual_seed(0)
    if cuda:
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = True

    # load < 2GB .mat files with scipy.io
    print('loading data mat file...')
    # mat = scipy.io.loadmat('/Users/glennjocher/Documents/PyCharmProjects/yolo/utils/class_chips48.mat')
    # X = np.ascontiguousarray(mat['X'])  # 596154x3x32x32
    # Y = np.ascontiguousarray(mat['Y'])

    # load > 2GB .mat files with h5py
    import h5py
    with h5py.File('/Users/glennjocher/Documents/PyCharmProjects/yolo/utils/class_chips64+64.mat') as mat:
        X = mat.get('X').value
        Y = mat.get('Y').value

    # # load with pickle
    # pickle.dump({'X': X, 'Y': Y}, open('save.p', "wb"), protocol=4)
    # with pickle.load(open('save.p', "rb")) as save:
    #     X, Y = save['X'], save['Y']

    X = np.ascontiguousarray(X)
    Y = np.ascontiguousarray(Y.ravel())

    # print('creating batches...')
    # train_data = create_batches(x=X, y=Y, batch_size=batch_size, shuffle=True)
    # del X, Y

    # Load saved model
    resume = False
    start_epoch = 0
    best_loss = float('inf')
    if resume:
        checkpoint = torch.load('best.pt', map_location='cuda:0' if cuda else 'cpu')

        model.load_state_dict(checkpoint['model'])

        # Set optimizer
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        optimizer.load_state_dict(checkpoint['optimizer'])

        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        del checkpoint
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model = model.to(device).train()
    weights = xview_class_weights(range(60))[Y].numpy()
    weights /= weights.sum()
    criteria = nn.CrossEntropyLoss()  # (weight=xview_class_weights(range(60)).to(device))
    stopper = patienceStopper(epochs=epochs, patience=patience, printerval=printerval)

    border = 32
    shape = X.shape[1:3]
    height = shape[0]
    print('training...', shape)

    def train(model):
        vC = torch.zeros(60).to(device)  # vector correct
        vS = torch.zeros(60).long().to(device)  # vecgtor samples
        loss_cum = torch.FloatTensor([0]).to(device)
        nS = len(Y)
        v = np.random.permutation(nS)
        for batch in range(int(nS / batch_size)):
            # i = v[batch * batch_size:(batch + 1) * batch_size]  # ordered chip selection
            i = np.random.choice(nS, size=batch_size, p=weights)  # weighted chip selection
            x, y = X[i], Y[i]

            # x = x.transpose([0, 2, 3, 1])  # torch to cv2
            for j in range(batch_size):
                M = random_affine(degrees=(-179, 179), translate=(.1, .1), scale=(.85, 1.20), shear=(-2, 2),
                                  shape=shape)

                x[j] = cv2.warpPerspective(x[j], M, dsize=shape, flags=cv2.INTER_AREA,
                                           borderValue=[60.134, 49.697, 40.746])  # RGB

            # import matplotlib.pyplot as plt
            # for pi in range(16):
            #     plt.subplot(4, 4, pi + 1).imshow(x[pi])
            # for pi in range(16):
            #    plt.subplot(4, 4, pi + 1).imshow(x[pi + 100, border:height - border, border:height - border])

            x = x.transpose([0, 3, 1, 2])  # cv2 to torch

            x = x[:, :, border:height - border, border:height - border]

            # if random.random() > 0.25:
            #     np.rot90(x, k=np.random.choice([1, 2, 3]), axes=(2, 3))
            # if random.random() > 0.5:
            #     x = x[:, :, :, ::-1]  # = np.fliplr(x)
            if random.random() > 0.5:
                x = x[:, :, ::-1, :]  # = np.flipud(x)

            # 596154x3x64x64
            # x_shift = int(np.clip(random.gauss(8, 3), a_min=0, a_max=16) + 0.5)
            # y_shift = int(np.clip(random.gauss(8, 3), a_min=0, a_max=16) + 0.5)
            # x = x[:, :, y_shift:y_shift + 48, x_shift:x_shift + 48]

            x = np.ascontiguousarray(x)
            x = torch.from_numpy(x).to(device).float()
            y = torch.from_numpy(y).to(device).long()

            x -= rgb_mean
            x /= rgb_std

            yhat = model(x)
            loss = criteria(yhat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                loss_cum += loss.data
                correct = y == torch.argmax(yhat.data, 1)
                vS += torch.bincount(y, minlength=60)
                vC += torch.bincount(y, minlength=60, weights=correct).float()

        accuracy = vC / vS.float()
        return loss_cum.detach().cpu(), accuracy.detach().cpu()

    for epoch in range(epochs):
        epoch += start_epoch
        loss, accuracy = train(model.train())

        # Save best checkpoint
        if (epoch >= 0) & (loss.item() < best_loss):
            best_loss = loss.item()
            torch.save({'epoch': epoch,
                        'best_loss': best_loss,
                        'accuracy': accuracy,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       'best64.pt')

        if stopper.step(loss, metrics=(*accuracy.mean().view(1),), model=model):
            break


def random_affine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-2, 2), shape=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    # Rotation and Scale
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    # a += random.choice([-180, -90, 0, 90])  # random 90deg rotations added to small rotations

    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(shape[1] / 2, shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * shape[0]  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * shape[1]  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

    M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
    return M


if __name__ == '__main__':
    main(ConvNetb())

# 64+64 chips, 3 layer, 64 filter, 1e-4 lr, weighted choice
# 14 layers, 4.30393e+06 parameters, 4.30393e+06 gradients
#        epoch        time        loss   metric(s)
#            0      60.166      753.99     0.22382
#            1      56.689       624.4     0.33007
#            2      57.275       582.1     0.36716
#            3      56.846      550.78      0.3957
#            4      57.729      527.38     0.41853
#            5      56.764      513.21     0.43129
#            6      56.875      498.57      0.4469
#            7      56.738      488.15     0.45739