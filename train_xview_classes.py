import math
import random
import argparse

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *

# sudo rm -rf mnist && git clone https://github.com/ultralytics/mnist && cd mnist && python3 train_xview_classes.py -run_name '10pad_6leaky_fullyconnected.pt'

parser = argparse.ArgumentParser()
parser.add_argument('-h5_name', default='../chips_10pad_square.h5', help='h5 filename')
parser.add_argument('-run_name', default='10pad_6ReLU_fullyconnected.pt', help='run name')
parser.add_argument('-resume', default=False, help='resume training flag')
opt = parser.parse_args()

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
            nn.Conv2d(3, n, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(n),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(n, n * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n * 2),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(n * 2, n * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n * 4),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(n * 4, n * 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n * 8),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(n * 8, n * 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n * 16),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(n * 16, n * 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n * 32),
            nn.ReLU())
        # self.layer7 = nn.Sequential(
        #     nn.Conv2d(n * 32, n * 64, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(n * 64),
        #     nn.ReLU())

        # self.fc = nn.Linear(int(8192), num_classes)  # 64 pixels, 4 layer, 64 filters
        self.fully_convolutional = nn.Conv2d(n * 32, 60, kernel_size=2, stride=1, padding=0, bias=True)

    def forward(self, x):  # 500 x 1 x 64 x 64
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        # x = self.layer7(x)
        # x = self.fc(x.reshape(x.size(0), -1))
        x = self.fully_convolutional(x)
        return x.squeeze()  # 500 x 60


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
    print('loading data...')
    # mat = scipy.io.loadmat('/Users/glennjocher/Documents/PyCharmProjects/yolo/utils/class_chips48.mat')
    # X = np.ascontiguousarray(mat['X'])  # 596154x3x32x32
    # Y = np.ascontiguousarray(mat['Y'])

    # load > 2GB .mat files with h5py
    import h5py
    with h5py.File(opt.h5_name) as h5:
        X = h5.get('X').value
        Y = h5.get('Y').value

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
    start_epoch = 0
    best_loss = float('inf')
    if opt.resume:
        checkpoint = torch.load(opt.run_name, map_location='cuda:0' if cuda else 'cpu')

        model.load_state_dict(checkpoint['model'])
        model = model.to(device).train()

        # Set optimizer
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        optimizer.load_state_dict(checkpoint['optimizer'])

        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        del checkpoint
    else:
        nGPU = torch.cuda.device_count()
        if nGPU > 1:
            print('%g GPUs found.' % nGPU)
            model = nn.DataParallel(model)
            model.to(device).train()
        else:
            model = model.to(device).train()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    weights = xview_class_weights(range(60))[Y].numpy()
    weights /= weights.sum()
    criteria = nn.CrossEntropyLoss()  # weight=xview_class_weights(range(60)).to(device))
    stopper = patienceStopper(epochs=epochs, patience=patience, printerval=printerval)

    border = 32
    shape = X.shape[1:3]
    height = shape[0]

    modelinfo(model)

    def train(model):
        vC = torch.zeros(60).to(device)  # vector correct
        vS = torch.zeros(60).long().to(device)  # vecgtor samples
        loss_cum = torch.FloatTensor([0]).to(device)
        nS = len(Y)
        # v = np.random.permutation(nS)
        for batch in range(int(nS / batch_size)):
            # i = v[batch * batch_size:(batch + 1) * batch_size]  # ordered chip selection
            i = np.random.choice(nS, size=batch_size, p=weights)  # weighted chip selection
            x, y = X[i], Y[i]

            # x = x.transpose([0, 2, 3, 1])  # torch to cv2
            for j in range(batch_size):
                M = random_affine(degrees=(-179.9, 179.9), translate=(.15, .15), scale=(.6, 1.40), shear=(-5, 5),
                                  shape=shape)

                x[j] = cv2.warpPerspective(x[j], M, dsize=shape, flags=cv2.INTER_LINEAR,
                                           borderValue=[60.134, 49.697, 40.746])  # RGB

            # import matplotlib.pyplot as plt
            # for pi in range(16):
            #     plt.subplot(4, 4, pi + 1).imshow(x[pi + 50])
            # for pi in range(16):
            #    plt.subplot(4, 4, pi + 1).imshow(x[pi + 50, border:height - border, border:height - border])

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
        if (epoch > 0) & (loss.item() < best_loss):
            best_loss = loss.item()
            torch.save({'epoch': epoch,
                        'best_loss': best_loss,
                        'accuracy': accuracy,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       opt.run_name)

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


def stripOptimizer():
    import torch
    a = torch.load('6leaky681.pt', map_location='cpu')
    a['optimizer'] = []
    torch.save(a, '6leaky681_stripped.pt')


if __name__ == '__main__':
    main(ConvNetb())

# Fully Convolutional
#                          name  gradient   parameters shape
# 15            layer6.0.weight      True  1.88744e+07 [2048, 1024, 3, 3]
# 16            layer6.1.weight      True         2048 [2048]
# 17              layer6.1.bias      True         2048 [2048]
# 18                  fc.weight      True       491520 [60, 2048, 2, 2]
# 19                    fc.bias      True           60 [60]

# Fully Connected
# 15            layer6.0.weight      True  1.88744e+07 [2048, 1024, 3, 3]
# 16            layer6.1.weight      True         2048 [2048]
# 17              layer6.1.bias      True         2048 [2048]
# 18                  fc.weight      True       491520 [60, 8192]
# 19                    fc.bias      True           60 [60]

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
#            8      57.036      475.83     0.46783
#            9      55.792      467.88     0.47626
#           10      56.208      458.48     0.48439
#           11      56.211      450.75     0.49385
#           12      57.053      445.68     0.49811
#           13      57.328      441.04     0.50464
#           14      56.918      431.16     0.51161
#           15      57.427      426.65     0.51633
#           16      57.459      419.86     0.52306
#           17      57.065      417.16     0.52744
#           18      56.941      412.04     0.52933
#           19      57.092       408.4     0.53467
#           20      56.203      405.08     0.53933

# 64+64 chips, 4 layer, 64 filter, 1e-4 lr, weighted choice
# 18 layers, 3.51904e+06 parameters, 3.51904e+06 gradients
#        epoch        time        loss   metric(s)
#            0      71.674      723.36     0.24818
#            1      68.146      578.31     0.36916
#            2      67.065      526.51     0.41884
#            3      65.809      489.59     0.45376
#            4      65.459       463.5     0.47846
#            5       66.26      444.56     0.49885
#            6      65.697       427.5     0.51586
#            7      66.678      411.46     0.52993
#            8      69.236      398.99     0.54557
#            9      67.304       387.8     0.55529
#           10       67.04      379.64     0.56469
#           11      68.929      366.64     0.57563
#           12      67.943      361.51     0.58113
#           13      67.129      351.83      0.5916
#           14      67.819      343.37     0.60065
#           15      66.663      336.71     0.60816
#           16      67.298      331.21     0.61232
#           17      66.624      327.19     0.61792
#           18      67.563      320.75     0.62496
#           19      66.685      314.04     0.63251
#           20      66.962      309.61     0.63594

# 64+64 chips, 5 layer, 64 filter, 1e-4 lr, weighted choice
# 22 layers, 7.25766e+06 parameters, 7.25766e+06 gradients
#        epoch        time        loss   metric(s)
#            0      82.027      716.17     0.25299
#            1       78.31      553.02     0.39201
#            2       77.94      494.01     0.44724
#            3      77.881      453.51     0.48681
#            4      78.541      422.42     0.51708
#            5      78.871      399.53      0.5412
#            6      79.004      380.04     0.56051
#            7      79.195      363.01      0.5776
#            8      79.192      348.36     0.59654
#            9      78.873       334.8     0.60685
#           10      78.701      325.81     0.62028
#           11      78.211      309.74     0.63352
#           12      78.383      304.03     0.64136
#           13      78.598      294.14      0.6517
#           14      78.995      284.86      0.6618
#           15      78.926      279.32     0.66773
#           16      79.018      272.16     0.67526
#           17      78.783       265.8     0.68253
#           18      79.131      258.86     0.69176
#           19      79.578      252.74     0.69823
#           20      79.602      248.09     0.70239

# 64+64 chips, 6 layer, 64 filter, 1e-4 lr, weighted choice
# 26 layers, 2.56467e+07 parameters, 2.56467e+07 gradients
#        epoch        time        loss   metric(s)
#            0      116.64      690.71     0.27556
#            1      112.58      519.11     0.42209
#            2      112.61      453.07      0.4859
#            3      111.94      405.52     0.53345
#            4      111.77       371.4      0.5683
#            5      111.45      346.25     0.59423
#            6      111.64      324.47     0.61758
#            7      111.63      303.77     0.63987
#            8      112.08      288.21     0.65884
#            9      112.17      275.39     0.67283
#           10      112.29      266.28     0.68319
#           11      111.44      251.77     0.69664
#           12      112.42      243.59     0.70702
#           13      112.55      234.84      0.7162
#           14      115.51      228.32     0.72272
#           15      115.35      219.51     0.73424
#           16      114.25       212.6     0.74147
#           17      111.66      208.52     0.74727
#           18      110.97       199.9     0.75598
#           19      111.33      196.14     0.76011
#           20      111.66      190.75     0.76805


# 64+64 chips, 6 layer, 64 filter, 1e-4 lr, weighted choice, higher augment, leakyRelu, 0% padding, fitted
#        epoch        time        loss   metric(s)
#            0      109.59      735.77     0.23381
#            1      108.29      586.93     0.35937
#            2      108.48      526.68      0.4152
#            3      108.85      480.95     0.45845
#            4      108.58      450.32     0.48716
#            5      119.43      424.43     0.51524
#            6      107.33       402.4     0.53763
#            7      107.14      384.57     0.55533
#            8      107.17       365.2     0.57515
#            9      107.13      352.65     0.59023

# 64+64 chips, 6 layer, 64 filter, 1e-4 lr, weighted choice, higher augment, leakyRelu, 20% padding, fitted
#        epoch        time        loss   metric(s)
#            0      108.85      725.42     0.23908
#            1      107.17      577.24     0.36326
#            2      107.04      515.71     0.42082
#            3      107.29      473.28     0.46295
#            4      107.16      440.75     0.49201
#            5      106.62       417.8     0.51934
#            6      106.91      396.42     0.53895
#            7       107.1      375.57     0.55968
#            8      107.02      360.21     0.57664
#            9      106.82       346.6     0.59196

# 64+64 chips, 6 layer, 64 filter, 1e-4 lr, weighted choice, higher augment, leakyRelu, 40% padding, fitted
#        epoch        time        loss   metric(s)
#            0      109.39      737.26     0.23002
#            1      106.75      584.28     0.35811
#            2      107.38      523.43     0.41684
#            3      107.93      480.15     0.45774
#            4      107.42      446.59     0.49093
#            5      107.53       421.9     0.51511
#            6      127.86      401.07     0.53674
#            7      107.33      381.23     0.55757
#            8      107.03      362.33     0.57763
#            9       107.7      350.94     0.59009

# 64+64 chips, 6 layer, 64 filter, 1e-4 lr, weighted choice, higher augment, leakyRelu, 0% padding, square
#        epoch        time        loss   metric(s)
#            0      112.74      743.18     0.22663
#            1      107.21      587.75     0.35784
#            2      107.58      524.38     0.41833
#            3      107.29       478.5     0.46096
#            4         107      446.26     0.49113
#            5       106.7      421.85     0.51618
#            6      107.62      399.23     0.53987
#            7      109.39      379.52     0.55968
#            8      108.83      363.36     0.57798
#            9      107.98      348.08     0.59201
#           10      107.88      337.13     0.60581
#           11      106.66      322.06     0.61837
#           12      106.85      313.77     0.62828
#           13      106.72      304.08     0.64047
#           14      106.69      293.42     0.64978
#           15      106.62      286.56     0.65798
#           16      106.61      277.47     0.66787
#           17      106.43      271.28     0.67516
#           18      106.86       264.2     0.68212
#           19      107.21      257.29     0.69141
#           20      107.13      251.61     0.69711

# 64+64 chips, 6 layer, 64 filter, 1e-4 lr, weighted choice, higher augment, leakyRelu, 10% padding, square
#        epoch        time        loss   metric(s)
#            0      110.02      735.73     0.23314
#            1       108.1      582.32     0.36034
#            2      108.12      519.97     0.42187
#            3      108.53      475.33     0.46494
#            4      108.45      440.62     0.49518
#            5      108.68      418.36     0.51985
#            6      109.23      395.31     0.54194
#            7      108.08      378.71     0.55991
#            8      108.09      359.13     0.57868
#            9      107.79      344.46      0.5949

# 64+64 chips, 6 layer, 64 filter, 1e-4 lr, weighted choice, higher augment, leakyRelu, 20% padding, square
#        epoch        time        loss   metric(s)
#            0      110.24       730.9     0.23271
#            1      107.62      577.26     0.36438
#            2      107.35       515.6      0.4228
#            3      107.62      472.82     0.46387
#            4      124.13      440.62     0.49341
#            5      107.23      417.91     0.51763
#            6      106.75      395.48     0.53978
#            7      106.99      377.17     0.55851
#            8      106.89      359.45     0.57787
#            9      106.77       345.5      0.5922

# 64+64 chips, 6 layer, 64 filter, 1e-4 lr, weighted choice, higher augment, leakyRelu, 30% padding, square
#        epoch        time        loss   metric(s)
#            0      109.46      733.33     0.23549
#            1      107.19      579.46     0.36461
#            2      106.98      516.34     0.42373
#            3      107.18      474.06      0.4635
#            4      107.19       441.6     0.49373
#            5      107.41      417.56     0.51907
#            6      107.36      393.83     0.54396
#            7       107.2      375.34     0.56278
#            8      106.94      357.22     0.58197
#            9      107.24      345.82     0.59314

# 64+64 chips, 6 layer, 64 filter, 1e-4 lr, weighted choice, higher augment, leakyRelu, 40% padding, square
#        epoch        time        loss   metric(s)
#            0      109.36      741.57     0.22691
#            1      107.26       586.7     0.35703
#            2      107.15      521.84     0.41817
#            3      107.53       480.9     0.45491
#            4      107.45      449.71      0.4861
#            5       107.2      424.02     0.51415
#            6      107.52      401.43     0.53515
#            7      108.87         383     0.55549
#            8      108.85      363.85     0.57416
#            9      108.63      348.33     0.59212

# 20% square normal ReLU
#        epoch        time        loss   metric(s)
#            0       108.9      737.91     0.22539
#            1      107.14      579.42     0.36284
#            2      106.87         516     0.42225
#            3      107.13      474.06     0.46322
#            4      107.28      442.12     0.49258
#            5      107.05      417.04      0.5179
#            6      107.19      394.21      0.5398
#            7      107.34      375.35     0.56192
#            8      106.97         356     0.58147
#            9      107.19      341.74     0.59599

# 20% square normal ReLU, Fully Convolutional
# epoch        time        loss   metric(s)
#            0      109.12      733.68     0.23079
#            1      107.11      572.85     0.36909
#            2      106.85      510.56     0.42809
#            3       106.5      467.99     0.46835
#            4      107.57      437.03     0.49939
#            5      107.72       409.7     0.52529
#            6       107.8      389.43     0.54693
#            7      107.55      370.69      0.5673
#            8      119.75      352.89     0.58503
#            9      107.21      341.62     0.59711

# winner ---> 20% square padding LeakyReLU ---> 7-layer (100M neurons!!!!)
# 23 layers, 1.00903e+08 parameters, 1.00903e+08 gradients
#        epoch        time        loss   metric(s)
#            0      217.05      700.71     0.26018
#            1       211.1         540     0.39818
#            2      210.54       476.1     0.45874
#            3      210.49      429.02     0.50559
#            4      210.54       394.6     0.53926
#            5      210.58      369.43     0.56643
#            6      211.34      347.43     0.58892
#            7       210.4      327.45     0.61295
#            8       210.3      307.95     0.63262
#            9      210.31      294.63     0.64601
