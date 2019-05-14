import scipy.io
import torch.nn as nn
from utils.utils import *
from models import *


def main(model):
    lr = .001
    epochs = 10
    printerval = 1
    patience = 200
    batch_size = 1000
    device = torch_utils.select_device()
    torch_utils.init_seeds()

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
