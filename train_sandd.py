# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import scipy.io

from models import *
from utils.utils import *


def main():
    """Train and evaluate the SANDD model on waveform data from a .mat file with specified hyperparameters."""
    lr = 0.001
    epochs = 10
    printerval = 1
    patience = 3
    batch_size = 1000
    device = torch_utils.select_device()
    torch_utils.init_seeds()

    mat = scipy.io.loadmat("data/sandd_training_data.mat")
    x = mat["waveforms"]  # inputs (nx512) [waveform1 waveform2]
    y = mat["targets"].ravel()  # outputs (nx4) [position(mm), time(ns), PE, E(MeV)]
    nz, nx = x.shape

    x, _, _ = normalize(x, 1)  # normalize each input row
    # y, ymu, ys = normalize(y, 0)  # normalize each output column
    x, y = torch.Tensor(x), torch.Tensor(y)
    x, y, xv, yv, xt, yt = split_data(x, y, train=0.70, validate=0.0, test=0.30, shuffle=True)

    train_loader = create_batches(x=x, y=y.squeeze().long(), batch_size=batch_size, shuffle=True)

    test_data = torch.Tensor(xt), torch.Tensor(yt).squeeze().long().to(device)
    # test_loader2 = create_batches(dataset=test_data, batch_size=10000)

    model = SANDD().to(device)
    criteria1 = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    stopper = patienceStopper(epochs=epochs, patience=patience, printerval=printerval)

    def train(model):
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            pred = model(x)

            loss = criteria1(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def test(model):
        x, y = test_data
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = criteria1(pred, y)
        yhat_number = torch.argmax(pred.data, 1)

        # prob = F.softmax(pred, 1)
        # plt.hist(prob[:, 1].detach(), 50)

        accuracy = []
        for i in range(2):
            j = y == i
            accuracy.append((yhat_number[j] == y[j]).float().mean() * 100.0)

        return loss, accuracy

    for epoch in range(epochs):
        train(model.train())
        loss, accuracy = test(model.eval())
        if stopper.step(loss, metrics=(*accuracy,), model=model):
            break


if __name__ == "__main__":
    main()
