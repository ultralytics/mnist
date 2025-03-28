# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import scipy.io

from models import *
from utils.utils import *

# import torchvision
# from torchvision import datasets, transforms


def main(model):
    """Trains and evaluates the given model on the MNIST dataset using custom training and testing loops."""
    lr = 0.001
    epochs = 20
    printerval = 1
    patience = 200
    batch_size = 1000
    device = torch_utils.select_device()
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

    # if not os.path.exists('data/MNISTtrain.mat'):
    #    scipy.io.savemat('data/MNISTtrain.mat',
    #                     {'x': train.train_data.unsqueeze(1).numpy(), 'y': train.train_labels.squeeze().numpy()})
    #    scipy.io.savemat('data/MNISTtest.mat',
    #                     {'x': test.test_data.unsqueeze(1).numpy(), 'y': test.test_labels.squeeze().numpy()})

    mat = scipy.io.loadmat("data/MNISTtrain.mat")
    train_loader2 = create_batches(
        x=torch.Tensor(mat["x"]), y=torch.Tensor(mat["y"]).squeeze().long(), batch_size=batch_size, shuffle=True
    )

    mat = scipy.io.loadmat("data/MNISTtest.mat")
    test_data = torch.Tensor(mat["x"]), torch.Tensor(mat["y"]).squeeze().long().to(device)
    # test_loader2 = create_batches(dataset=test_data, batch_size=10000)

    model = model.to(device)
    criteria1 = nn.CrossEntropyLoss()
    criteria2 = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    stopper = patienceStopper(epochs=epochs, patience=patience, printerval=printerval)

    def train(model):
        for i, (x, y) in enumerate(train_loader2):
            x, y = x.to(device), y.to(device)
            # x = x.repeat([1, 3, 1, 1])  # grey to rgb
            pred = model(x)

            y2 = torch.zeros_like(pred)
            for j in range(len(pred)):
                y2[j, y[j]] = 1

            loss = criteria2(pred, y2)

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


if __name__ == "__main__":
    # model=MLP()
    # model = ConvNeta()
    model = ConvNetb()

    # # load pretrained model: https://github.com/Cadene/pretrained-models.pytorch#torchvision
    # import pretrainedmodels
    # model_name = 'resnet101'
    # model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    #
    # # adjust last layer
    # n = 10  # desired classes
    # filters = model.last_linear.weight.shape[1]
    # model.last_linear.bias = torch.nn.Parameter(torch.zeros(n))
    # model.last_linear.weight = torch.nn.Parameter(torch.zeros(n, filters))
    # model.last_linear.out_features = n

    # Train
    torch_utils.model_info(model, report="full")
    main(model)
