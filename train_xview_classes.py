# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import argparse
import math

import cv2
import torch.nn as nn

from utils.utils import *

# Start New Training
# sudo rm -rf mnist && git clone https://github.com/ultralytics/mnist && cd mnist && python3 train_xview_classes.py -run_name '5leaky64.pt'

# Resume Training
# cd mnist && python3 train_xview_classes.py -run_name '10pad_64f_5leaky.pt' -resume 1

parser = argparse.ArgumentParser()
parser.add_argument("-h5_name", default="../chips_10pad_square.h5", help="h5 filename")
parser.add_argument("-run_name", default="10pad_64f_5leaky.pt", help="run name")
parser.add_argument("-resume", default=False, help="resume training flag")
opt = parser.parse_args()
print(opt)


def xview_class_weights(indices):  # weights of each class in the training set, normalized to mu = 1
    """Compute and return the normalized class weights for given indices in the xView dataset training set."""
    weights = 1 / torch.FloatTensor(
        [
            74,
            364,
            713,
            71,
            2925,
            20976.7,
            6925,
            1101,
            3612,
            12134,
            5871,
            3640,
            860,
            4062,
            895,
            149,
            174,
            17,
            1624,
            1846,
            125,
            122,
            124,
            662,
            1452,
            697,
            222,
            190,
            786,
            200,
            450,
            295,
            79,
            205,
            156,
            181,
            70,
            64,
            337,
            1352,
            336,
            78,
            628,
            841,
            287,
            83,
            702,
            1177,
            31386.5,
            195,
            1081,
            882,
            1059,
            4175,
            123,
            1700,
            2317,
            1579,
            368,
            85,
        ]
    )
    weights /= weights.sum()
    return weights[indices]


# https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/02-intermediate
class ConvNetb(nn.Module):
    """A CNN model for image classification with Conv2d, BatchNorm2d, and LeakyReLU layers."""

    def __init__(self, num_classes=60):
        """Initializes the ConvNetb model with convolutional, batch normalization, and LeakyReLU layers, setting the
        number of classes.
        """
        super().__init__()
        n = 64  # initial convolution size
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, n, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(n), nn.LeakyReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(n, n * 2, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(n * 2), nn.LeakyReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(n * 2, n * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n * 4),
            nn.LeakyReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(n * 4, n * 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n * 8),
            nn.LeakyReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(n * 8, n * 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n * 16),
            nn.LeakyReLU(),
        )
        # self.layer6 = nn.Sequential(
        #     nn.Conv2d(n * 16, n * 32, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(n * 32),
        #     nn.LeakyReLU(),
        #     nn.Dropout2d(0.5))

        # self.fc = nn.Linear(int(8192/2), num_classes)  # 64 pixels, 4 layer, 64 filters
        self.fully_conv = nn.Conv2d(n * 16, num_classes, kernel_size=4, stride=1, padding=0, bias=True)  # 5 layer s1

    def forward(self, x):  # 500 x 1 x 64 x 64
        """Performs a forward pass through the network with input tensor x of shape (500 x 1 x 64 x 64)."""
        # xa = [x]  # 0 rot
        # xa.append(x.flip(2).flip(3))  # 180 rot
        # xa.append(x.transpose(2, 3).flip(2))  # -90 rot
        # xa.append(x.transpose(2, 3).flip(3))  # +90 rot
        #
        # # import matplotlib.pyplot as plt
        # # plt.subplot(2, 2, 1).imshow(b[0][1, 1].detach().numpy())
        # # plt.subplot(2, 2, 2).imshow(b[1][1, 1].detach().numpy())
        # # plt.subplot(2, 2, 3).imshow(b[2][1, 1].detach().numpy())
        # # plt.subplot(2, 2, 4).imshow(b[3][1, 1].detach().numpy())
        #
        # v, b = [], []
        # for i, a in enumerate(xa):
        #     b.append(self.layer1(a))
        #     v.append(b[i].sum(3).sum(2).sum(1).unsqueeze(1))
        # v = torch.cat(v, 1)
        #
        # best_transform_index = torch.argmax(v, 1)
        #
        # x = b[0]
        # for i, bt in enumerate(best_transform_index):
        #     if bt>0:
        #         x[i] = b[bt][i]

        x = self.layer1(x)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x = self.layer4(x)
        # print(x.shape)
        x = self.layer5(x)
        # print(x.shape)
        # x = self.layer6(x)
        # x = self.fc(x.reshape(x.size(0), -1))
        x = self.fully_conv(x)
        return x.squeeze()  # 500 x 60


# @profile
def main(model):
    """Execute training and testing loop for the model with dataset loading, preprocessing, and checkpoint saving."""
    lr = 0.0001
    epochs = 1000
    printerval = 1
    patience = 500
    batch_size = 100
    device = torch_utils.select_device()
    torch_utils.init_seeds()

    rgb_mean = torch.FloatTensor([60.134, 49.697, 40.746]).view((1, 3, 1, 1)).to(device)
    rgb_std = torch.FloatTensor([29.99, 24.498, 22.046]).view((1, 3, 1, 1)).to(device)

    # load < 2GB .mat files with scipy.io
    print("loading data...")
    # mat = scipy.io.loadmat('/Users/glennjocher/Documents/PyCharmProjects/yolo/utils/class_chips48.mat')
    # X = np.ascontiguousarray(mat['X'])  # 596154x3x32x32
    # Y = np.ascontiguousarray(mat['Y'])

    # load > 2GB .mat files with h5py
    import h5py

    with h5py.File(opt.h5_name) as h5:
        X = h5.get("X").value
        Y = h5.get("Y").value

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
    best_loss = float("inf")
    nGPU = torch.cuda.device_count()
    if opt.resume:
        checkpoint = torch.load(opt.run_name, map_location=device)

        model.load_state_dict(checkpoint["model"])
        if nGPU > 1:
            print(f"{nGPU:g} GPUs found.")
            model = nn.DataParallel(model)
        model.to(device).train()

        # Set optimizer
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(checkpoint["optimizer"])

        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]
        del checkpoint
    else:
        if nGPU > 1:
            print(f"{nGPU:g} GPUs found.")
            model = nn.DataParallel(model)
        model.to(device).train()

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=5e-4)

    # Split data into train and test groups
    weights = xview_class_weights(range(60))[Y].numpy()
    weights /= weights.sum()

    nS = len(Y)
    mask = np.zeros(nS)
    for i in range(60):
        j = np.nonzero(Y == i)[0]
        n = len(j)
        mask[j[np.random.choice(n, size=int(n * 0.1), replace=False)]] = 1

    mask = mask == 1
    X_test, Y_test = X[mask].copy(), Y[mask].copy()
    X, Y, weights = X[~mask], Y[~mask], weights[~mask]
    weights /= weights.sum()

    X = np.ascontiguousarray(X)
    Y = np.ascontiguousarray(Y.ravel())
    X_test = np.ascontiguousarray(X_test)
    Y_test = np.ascontiguousarray(Y_test.ravel())

    criteria = nn.CrossEntropyLoss()  # weight=xview_class_weights(range(60)).to(device))
    stopper = patienceStopper(epochs=epochs, patience=patience, printerval=printerval)
    model_info(model)

    border = 32
    shape = X.shape[1:3]

    def train(model):
        vC = torch.zeros(60).to(device)  # vector correct
        vS = torch.zeros(60).long().to(device)  # vector samples
        loss_cum = torch.FloatTensor([0]).to(device)
        nS = len(Y)
        # v = np.random.permutation(nS)
        for batch in range(int(nS / batch_size)):
            # i = v[batch * batch_size:(batch + 1) * batch_size]  # ordered chip selection
            i = np.random.choice(nS, size=batch_size, p=weights)  # weighted chip selection
            x, y = X[i], Y[i]

            # x = x.transpose([0, 2, 3, 1])  # torch to cv2
            for j in range(batch_size):
                augment_hsv = False
                if augment_hsv:
                    # SV augmentation by 50%
                    fraction = 0.50
                    img_hsv = cv2.cvtColor(x[j], cv2.COLOR_RGB2HSV)
                    S = img_hsv[:, :, 1].astype(np.float32)
                    V = img_hsv[:, :, 2].astype(np.float32)

                    a = (random.random() * 2 - 1) * fraction + 1
                    S *= a
                    if a > 1:
                        np.clip(S, a_min=0, a_max=255, out=S)

                    a = (random.random() * 2 - 1) * fraction + 1
                    V *= a
                    if a > 1:
                        np.clip(V, a_min=0, a_max=255, out=V)

                    img_hsv[:, :, 1] = S.astype(np.uint8)
                    img_hsv[:, :, 2] = V.astype(np.uint8)
                    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB, dst=x[j])

                M = random_affine(
                    degrees=(-179.9, 179.9), translate=(0.15, 0.15), scale=(0.75, 1.40), shear=(-3, 3), shape=shape
                )

                x[j] = cv2.warpPerspective(x[j], M, dsize=shape, flags=cv2.INTER_LINEAR)
                # borderValue=[60.134, 49.697, 40.746])  # RGB

                if random.random() > 0.5:
                    x[j] = x[j, ::-1]  # = np.flipud(x)

            # import matplotlib.pyplot as plt
            # plt.hist(Y[i],60)
            # for pi in range(16):
            #     plt.subplot(4, 4, pi + 1).imshow(x[pi + 50])
            # for pi in range(16):
            #     plt.subplot(4, 4, pi + 1).imshow(x[pi + 50, border:-border, border:-border])

            x = x[:, border:-border, border:-border]

            x = x.transpose([0, 3, 1, 2])  # cv2 to torch

            # if random.random() > 0.25:
            #     np.rot90(x, k=np.random.choice([1, 2, 3]), axes=(2, 3))
            # if random.random() > 0.5:
            #     x = x[:, :, :, ::-1]  # = np.fliplr(x)
            # if random.random() > 0.5:
            #    x = x[:, :, ::-1, :]  # = np.flipud(x)

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

    def test(model):
        vC = torch.zeros(60).to(device)  # vector correct
        vS = torch.zeros(60).long().to(device)  # vector samples
        loss_cum = torch.FloatTensor([0]).to(device)
        nS = len(Y_test)
        v = np.random.permutation(nS)
        for batch in range(int(nS / batch_size)):
            i = v[batch * batch_size : np.minimum((batch + 1) * batch_size, nS)]  # ordered chip selection
            # i = np.random.choice(nS, size=batch_size, p=weights)  # weighted chip selection
            x, y = X_test[i], Y_test[i]

            x = x[:, border:-border, border:-border]

            x = x.transpose([0, 3, 1, 2])  # cv2 to torch

            x = np.ascontiguousarray(x)
            x = torch.from_numpy(x).to(device).float()
            y = torch.from_numpy(y).to(device).long()

            x -= rgb_mean
            x /= rgb_std

            with torch.no_grad():
                yhat = model(x)
                loss = criteria(yhat, y)

                loss_cum += loss.data
                correct = y == torch.argmax(yhat.data, 1)
                vS += torch.bincount(y, minlength=60)
                vC += torch.bincount(y, minlength=60, weights=correct).float()

        accuracy = vC / vS.float()
        return loss_cum.detach().cpu(), accuracy.detach().cpu()

    for epoch in range(epochs):
        epoch += start_epoch
        loss, accuracy = train(model.train())
        loss_test, accuracy_test = test(model.eval())

        # Save best checkpoint
        if (epoch > 0) & (loss_test.item() < best_loss):
            best_loss = loss_test.item()
            torch.save(
                {
                    "epoch": epoch,
                    "best_loss": best_loss,
                    "accuracy": accuracy_test,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                opt.run_name,
            )

        if stopper.step(
            loss,
            metrics=(
                *accuracy.mean().view(1),
                loss_test,
                *accuracy_test.mean().view(1),
            ),
            model=model,
        ):
            break


def random_affine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-2, 2), shape=(0, 0)):
    """Apply random affine transformations including rotation, translation, scaling, and shearing to an image with a
    given shape.
    """
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

    return S @ T @ R  # ORDER IS IMPORTANT HERE!!


def strip_optimizer_from_checkpoint(filename="checkpoints/best.pt"):
    """Removes the optimizer from the checkpoint file to reduce its size and saves the modified checkpoint."""
    import torch

    a = torch.load(filename, map_location="cpu")
    a["optimizer"] = []
    torch.save(a, filename.replace(".pt", "_lite.pt"))


if __name__ == "__main__":
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

# 5 layer leaky SV+spatial augment, 64+64 pixels, 500 bs
# 17 layers, 7.25568e+06 parameters, 7.25568e+06 gradients
#        epoch        time        loss   metric(s)
#            0      122.51      745.02     0.14759      80.922     0.26851
#            1      121.55      637.62     0.24055       68.51     0.34984
#            2      121.14      589.01      0.2882      68.981       0.373
#            3      121.96      553.68     0.32427      71.004     0.38597
#            4      121.25       527.6     0.35055      63.153      0.4134
#            5      120.76      500.13     0.38169      65.824     0.45979
#            6      120.96       479.2     0.40363      64.058     0.45603
#            7      121.18      464.43     0.41856       63.81      0.4781
#            8      121.24      448.43     0.43534      59.518      0.4814
#            9       121.3      436.32     0.45119      59.821     0.49955
#           10      121.78      424.26     0.46264       57.99     0.50135

# 5 layer leaky SV+spatial augment, 64+64 pixels, 100 bs
# 17 layers, 7.25568e+06 parameters, 7.25568e+06 gradients
#        epoch        time        loss   metric(s)
#            0      128.67      3509.1      0.1836      384.25     0.30715
#            1      128.56      2937.2     0.28962      337.31     0.40175
#            2      127.99      2637.8     0.34943      326.57     0.42516
#            3      127.98      2459.5     0.38902      306.14     0.44886
#            4      128.08      2318.4     0.41683      333.77     0.46359
#            5      127.72      2206.6     0.44108      266.26     0.50366
#            6      127.43      2111.6     0.46424      284.99     0.50487

# 5 layer leaky SV+spatial augment, 64+64 pixels, 100 bs + best 90deg rot
# 17 layers, 7.25568e+06 parameters, 7.25568e+06 gradients
#        epoch        time        loss   metric(s)
#            0      288.12      3428.7     0.20071      396.57     0.34167
#            1       287.9      2799.2     0.31593      326.79      0.4228
#            2       287.9      2522.7     0.37527      316.94     0.45251
#            3      287.91        2352     0.41048      287.74       0.492
#            4      287.94      2225.1     0.43963      308.38     0.50274
#            5      287.99        2108     0.46444      275.92       0.504
#            6      288.14      2024.6     0.48631      296.98     0.49741
