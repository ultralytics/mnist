# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import glob
import os

import cv2
from tqdm import tqdm

from models import *
from utils.utils import *


def main(model):
    """Trains a model on a binary classifier dataset and evaluates it, saving the best model state to Google Cloud
    Storage.
    """
    lr = 0.0005
    epochs = 10
    printerval = 1
    patience = 5
    batch_size = 64
    device = torch_utils.select_device(device="0")
    torch_utils.init_seeds()

    # MNIST Dataset
    # tforms = transforms.Compose([torchvision.transforms.RandomAffine(
    #    degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10)),
    #    transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # tformstest = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # train = datasets.MNIST(root='./data', train=True, transform=tforms, download=True)
    # test = datasets.MNIST(root='./data', train=False, transform=tformstest)
    # train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=False)
    # test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=10000, shuffle=False)

    # binary classifier dataset
    path = "../knife_classifier/"
    d = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]  # category directories

    x, y = [], []
    for i, c in enumerate(d):
        for file in tqdm(glob.glob(f"{c}/*.*")[:9000]):
            img = cv2.resize(cv2.imread(file), (128, 128))  # BGR
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.expand_dims(img, axis=0)  # add batch dim
            img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0

            x.append(img)  # input
            y.append(i)  # output

    print("Concatenating...")
    x = np.concatenate(x, 0)
    y = np.array(y)
    nc = len(np.unique(y))  # number of classes

    print("Splitting into train and validate sets...")
    x, y, xtest, ytest, *_ = split_data(x, y, train=0.8, validate=0.20, test=0.0, shuffle=True)

    print("Creating Train Dataloader...")
    train_loader = create_batches(
        x=torch.Tensor(x),  # [60000, 1, 28, 28]
        y=torch.Tensor(y).squeeze().long(),  # [60000]
        batch_size=batch_size,
        shuffle=True,
    )
    del x, y

    print("Creating Test Dataloader...")
    test_loader = create_batches(x=torch.Tensor(xtest), y=torch.Tensor(ytest).squeeze().long(), batch_size=batch_size)
    del xtest, ytest

    # import scipy.io
    # if not os.path.exists('data/MNISTtrain.mat'):
    #    scipy.io.savemat('data/MNISTtrain.mat',
    #                     {'x': train.train_data.unsqueeze(1).numpy(), 'y': train.train_labels.squeeze().numpy()})
    #    scipy.io.savemat('data/MNISTtest.mat',
    #                     {'x': test.test_data.unsqueeze(1).numpy(), 'y': test.test_labels.squeeze().numpy()})

    # mat = scipy.io.loadmat('data/MNISTtrain.mat')
    # train_loader = create_batches(x=torch.Tensor(mat['x']),  # [60000, 1, 28, 28]
    #                                y=torch.Tensor(mat['y']).squeeze().long(),  # [60000]
    #                                batch_size=batch_size, shuffle=True)
    #
    # mat = scipy.io.loadmat('data/MNISTtest.mat')
    # # test_data = torch.Tensor(mat['x']), torch.Tensor(mat['y']).squeeze().long().to(device)
    # test_loader = create_batches(x=torch.Tensor(mat['x']),
    #                               y=torch.Tensor(mat['y']).squeeze().long(),
    #                               batch_size=batch_size)

    model = model.to(device)
    criteria = nn.CrossEntropyLoss()
    # criteria2 = nn.BCEWithLogitsLoss()

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.90, weight_decay=1e-5, nesterov=True)
    stopper = patienceStopper(epochs=epochs, patience=patience, printerval=printerval)

    print("Starting training...")

    def train(model):
        pbar = tqdm(enumerate(train_loader), desc="train", total=len(train_loader))  # progress bar
        for i, (x, y) in pbar:
            x, y = x.to(device), y.to(device)
            # x = x.repeat([1, 3, 1, 1])  # grey to rgb
            # x /= 255.  # rescale to 0-1

            augment = True
            if augment:
                # random left-right flip
                lr_flip = True
                if lr_flip and random.random() < 0.5:
                    x = torch.flip(x, [3])

                # random up-down flip
                ud_flip = True
                if ud_flip and random.random() < 0.5:
                    x = torch.flip(x, [2])

            loss = criteria(model(x), y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def test(model):
        pbar = tqdm(enumerate(test_loader), desc="test", total=len(test_loader))  # progress bar
        for i, (x, y) in pbar:
            x, y = x.to(device), y.to(device)
            # x = x.repeat([1, 3, 1, 1])  # grey to rgb
            # x /= 255.  # rescale to 0-1

            pred = model(x)
            loss = criteria(pred, y)

            accuracy = []
            pred_class = torch.argmax(pred.data, 1)
            for c in range(nc):
                j = y == c
                accuracy.append((pred_class[j] == y[j]).float().mean() * 100.0)

        return loss, accuracy

    for epoch in range(epochs):
        train(model.train())
        loss, accuracy = test(model.eval())
        if stopper.step(loss, metrics=(*accuracy,), model=model):
            break

    # save model
    f = "resnet101.pt"
    bucket = "yolov4"
    chkpt = {"model": stopper.bestmodel.state_dict()}
    torch.save(chkpt, f)
    os.system(f"gsutil cp -r {f} gs://{bucket}")


if __name__ == "__main__":
    # model=MLP()
    # model = ConvNeta()
    # model = ConvNetb()
    model = torch_utils.load_classifier(name="resnet101", n=2)

    # Train
    main(model)
