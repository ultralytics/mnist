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
        n = 32  # initial convolution size
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
        # self.fc = nn.Linear(65536, num_classes)  # 64+36, 3layer 64n
        self.fc = nn.Linear(32768, num_classes)  # 64+36, 3layer 32n

    def forward(self, x):  # x.size() = [512, 1, 28, 28]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        # x = self.layer5(x)
        x = x.reshape(x.size(0), -1)
        # x, _, _ = normalize(x,1)
        x = self.fc(x)
        # x = torch.sigmoid(x)
        # x = F.log_softmax(x, dim=1)  # ONLY for use with nn.NLLLoss
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
                       'best.pt')

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

# 46 3layer 32n
# ﻿14 layers, 585276 parameters, 585276 gradients
#        epoch        time        loss   metric(s)
#            0      56.311        2062     0.13002
#            1      50.566      1753.9     0.22344
#            2      50.651      1626.5     0.27488
#            3      50.375      1531.5     0.31341
#            4       50.17      1485.3     0.32498
#            5      50.143      1439.5     0.35106
#            6      49.461      1390.4      0.3626
#            7      48.164      1356.6     0.38024
#            8      50.116      1311.9     0.39458
#            9      50.105        1287     0.40788

# 64 3layer 32n
# 14 layers, 1.19968e+06 parameters, 1.19968e+06 gradients
#        epoch        time        loss   metric(s)
#            0      112.31      2069.3     0.13801
#            1      101.42      1703.3     0.23556
#            2      98.379      1580.4     0.29431
#            3      94.615      1489.7     0.32182
#            4      94.348      1451.7     0.33702
#            5      94.186        1408     0.35843
#            6      94.337      1345.2     0.37453
#            7      94.552      1314.4     0.39064
#            8      94.547      1281.4     0.40956
#            9       94.64      1249.2     0.42411

# 64 3layer 64n
# 14 layers, 2.58361e+06 parameters, 2.58361e+06 gradients
#        epoch        time        loss   metric(s)
#            0      151.47      2227.2     0.12849
#            1      138.06      1714.7     0.24045
#            2      137.25      1571.3     0.29142
#            3      137.39      1482.8     0.32846
#            4      139.43      1436.2     0.34547
#            5      139.55      1383.7     0.36446
#            6       140.1      1316.7      0.3892
#            7      139.44      1284.8     0.40241
# updated lr 0.001 bs 1000
# 14 layers, 2.58361e+06 parameters, 2.58361e+06 gradients
#        epoch        time        loss   metric(s)
#            0      150.14        2111     0.12251  0.0001 lr
#            0      150.72      2399.9    0.089126
#            1      143.15      1896.3     0.17328
#            2       144.2      1757.8     0.22221
#            3      144.11      1678.7     0.24831
#            4      142.85      1615.2     0.28278
#            5      142.18      1558.4     0.29893
#            6       142.2      1524.2     0.31689
#            7       142.2      1484.6     0.32971
#            8      142.05      1448.5     0.34189
#            9      141.92      1417.2     0.35459
#           10      144.75      1388.6     0.36825


# 64 4layer 16n
#            0      83.699      2041.6     0.13639
#            1       80.72      1744.9     0.22675
#            2      81.325      1620.5     0.27641
#            3      80.172      1543.8     0.29958
#            4      80.119      1498.4     0.32014
#            5      79.339      1459.3     0.34453
#            6      79.586      1418.1     0.35016
#            7      79.298      1389.8     0.36621
#            8      77.316        1353     0.37321
#            9       73.95      1323.8     0.38569

#                            name  gradient   parameters shape
#    0            layer1.0.weight      True          864 [32, 3, 3, 3]
#    1              layer1.0.bias      True           32 [32]
#    2            layer1.1.weight      True           32 [32]
#    3              layer1.1.bias      True           32 [32]
#    4            layer2.0.weight      True        18432 [64, 32, 3, 3]
#    5              layer2.0.bias      True           64 [64]
#    6            layer2.1.weight      True           64 [64]
#    7              layer2.1.bias      True           64 [64]
#    8            layer3.0.weight      True        73728 [128, 64, 3, 3]
#    9              layer3.0.bias      True          128 [128]
#   10            layer3.1.weight      True          128 [128]
#   11              layer3.1.bias      True          128 [128]
#   12                  fc.weight      True       491520 [60, 8192]
#   13                    fc.bias      True           60 [60]
#
# 14 layers, 585276 parameters, 585276 gradients
#        epoch        time        loss   metric(s)
#            0      20.395      1924.2     0.18857
#            1      19.423      1503.9     0.33079
#            2      19.487      1319.7     0.40424
#            3      19.681      1191.8     0.45266
#            4      19.709      1096.1     0.49585
#            5      19.816      1029.6     0.52194
#            6      19.762      946.29     0.56495
#            7      19.783      888.56     0.58585
#            8      19.724      827.64     0.61555
#            9      19.727      798.74     0.62834
#           10      19.911      743.75     0.65416
#           11      19.944      697.44     0.67834
#           12      19.868      686.78     0.67996
#           13      19.864      642.56     0.70371
#           14      20.028      620.39     0.70885
#           15      19.877      581.02     0.73049
#           16      19.911      568.15     0.73423
#           17      19.912      547.26      0.7433
#           18      19.949      526.46     0.75086
#           19      20.444      509.22     0.76147
#           20      19.894      494.67      0.7712
#           21      19.857      475.11     0.77759
#           22      19.925      454.83     0.78423
#           23      19.896      456.91     0.78415
#           24      19.879      457.18      0.7838
#           25      19.881      425.15     0.79752
#           26      19.914      407.88     0.80742
#           27      19.873      407.97     0.80539
#           28      19.839      413.96     0.80124
#           29       19.82      397.62     0.80809
#           30       19.88       390.3     0.81389
#           31      19.924      377.53     0.81872
#           32      19.911      381.52     0.81722
#           33      20.223       364.2     0.82516
#           34      20.191      360.64     0.82682
#           35      20.218      355.63     0.82859
#           36      20.168      357.96     0.83001
#           37      20.146      334.97     0.83944
#           38       20.13      329.62     0.84092
#           39      20.069      331.98     0.84127
#           40       20.11      320.05      0.8451
#           41      20.269      315.99     0.84778
#           42      20.521       305.5     0.85305
#           43      20.868      311.85     0.84961
#           44      20.766      305.16     0.85248
#           45      20.393      310.13     0.85014
#           46      20.229      305.05     0.85204
#           47      20.177       297.3     0.85537
#           48      20.356      292.04     0.85795
#           49      20.185      294.98     0.85551
#           50      20.087      284.86     0.86022
#           51      20.085      277.81     0.86635
#           52      19.958      277.42     0.86377
#           53      20.037      274.32     0.86741
#           54      20.018      283.44     0.86148
#           55      20.027      277.45     0.86231
#           56      20.064      266.32     0.87014
#           57      20.173      268.89       0.869
#           58      20.099      266.23     0.86944
#           59      20.015      273.43     0.86518
#           60      20.071      261.52      0.8715
#           61      19.932      257.48     0.87168
#           62      20.133      247.77     0.87815
#           63      20.435      252.39     0.87666
#           64      25.582      249.13      0.8782
#           65       21.96      256.82     0.87189


#                            name  gradient   parameters shape
#    0            layer1.0.weight      True         1296 [48, 3, 3, 3]
#    1              layer1.0.bias      True           48 [48]
#    2            layer1.1.weight      True           48 [48]
#    3              layer1.1.bias      True           48 [48]
#    4            layer2.0.weight      True        41472 [96, 48, 3, 3]
#    5              layer2.0.bias      True           96 [96]
#    6            layer2.1.weight      True           96 [96]
#    7              layer2.1.bias      True           96 [96]
#    8            layer3.0.weight      True       165888 [192, 96, 3, 3]
#    9              layer3.0.bias      True          192 [192]
#   10            layer3.1.weight      True          192 [192]
#   11              layer3.1.bias      True          192 [192]
#   12                  fc.weight      True  1.65888e+06 [60, 27648]
#   13                    fc.bias      True           60 [60]
#
# 14 layers, 1.8686e+06 parameters, 1.8686e+06 gradients
#        epoch        time        loss   metric(s)
#            0      70.877      2059.2      0.1769
#            1      70.276      1480.9     0.34159
#            2      70.269      1276.3     0.42205
#            3      70.509      1147.6     0.48354
#            4      70.773      1037.7     0.52214
#            5      70.863       925.2     0.57857
#            6      70.571      876.61     0.60539
#            7      70.895      804.31     0.62645
#            8      70.763      718.81     0.66836
#            9      70.505      687.02     0.68107
#           10       70.72      617.72     0.71195
#           11      70.837      598.75     0.72256
#           12      70.695      527.74     0.75268
#           13       70.64      514.62     0.76154
#           14      70.553      471.66      0.7775
#           15      70.581      459.28      0.7863
#           16      70.439      429.88     0.79851
#           17       70.49      365.05     0.82842
#           18      70.408      379.39     0.82186
#           19      70.399      368.75     0.82778
#           20      70.505      343.13     0.83721
#           21        70.5      324.47     0.84568
#           22      70.581      322.25      0.8466
#           23      70.718      303.18     0.85405
#           24      70.461      288.43     0.86099
#           25      70.495      275.71     0.86802
#           26      70.356      266.83     0.87189
#           27      70.363      254.95     0.87826
#           28      70.491      255.55     0.87825
#           29      70.812      248.45     0.87903
#           30      71.061      238.48     0.88388
#           31      70.678      231.15      0.8889
#           32       70.73      225.54     0.89089
#           33      70.592      210.93     0.89717
#           34      70.817      212.87     0.89526
#           35      70.481      210.58     0.89769
#           36      70.601      211.95     0.89671
#           37       70.73      192.43     0.90523
#           38      70.726      192.81     0.90397
#           39      70.663      193.53     0.90412
#           40      70.561      184.37     0.90856
#           41       70.57         179     0.91103
#           42      71.008      184.24     0.90801
#           43      70.923      174.11     0.91435
#           44      70.652      170.29      0.9153
#           45      70.477      168.18     0.91574
#           46      70.591      158.62     0.92013
#           47      70.631      155.97     0.92199
#           48      70.802      154.02     0.92242
#           49       70.62      159.06     0.92066
