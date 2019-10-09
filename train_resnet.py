import glob
import os

import cv2
from tqdm import tqdm

from models import *
from utils.utils import *


def main(model):
    lr = 0.01
    epochs = 1000
    printerval = 1
    patience = 100
    batch_size = 32
    device = torch_utils.select_device(device='1' if torch.cuda.is_available() else '')
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
    path = '../knife_classifier/'
    d = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]  # category directories

    x, y = [], []
    for i, c in enumerate(d):
        for file in tqdm(glob.glob('%s/*.*' % c)[:1000]):
            img = cv2.resize(cv2.imread(file), (64, 64))  # BGR
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.expand_dims(img, axis=0)  # add batch dim
            img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0

            x.append(img)  # input
            y.append(i)  # output

    print('Concatenating...')
    x = np.concatenate(x, 0)
    y = np.array(y)
    nc = len(np.unique(y))  # number of classes

    print('Shuffling...')
    shuffle_data(x, y)

    print('Splitting into train and validate sets...')
    x, y, xtest, ytest, *_ = split_data(x, y, train=0.9, validate=0.10, test=0.0)

    print('Creating Train Dataloader...')
    train_loader2 = create_batches(x=torch.Tensor(x),  # [60000, 1, 28, 28]
                                   y=torch.Tensor(y).squeeze().long(),  # [60000]
                                   batch_size=batch_size, shuffle=True)
    del x, y

    print('Creating Test Dataloader...')
    test_loader2 = create_batches(x=torch.Tensor(xtest),
                                  y=torch.Tensor(ytest).squeeze().long(),
                                  batch_size=batch_size)
    del xtest, ytest

    # import scipy.io
    # if not os.path.exists('data/MNISTtrain.mat'):
    #    scipy.io.savemat('data/MNISTtrain.mat',
    #                     {'x': train.train_data.unsqueeze(1).numpy(), 'y': train.train_labels.squeeze().numpy()})
    #    scipy.io.savemat('data/MNISTtest.mat',
    #                     {'x': test.test_data.unsqueeze(1).numpy(), 'y': test.test_labels.squeeze().numpy()})

    # mat = scipy.io.loadmat('data/MNISTtrain.mat')
    # train_loader2 = create_batches(x=torch.Tensor(mat['x']),  # [60000, 1, 28, 28]
    #                                y=torch.Tensor(mat['y']).squeeze().long(),  # [60000]
    #                                batch_size=batch_size, shuffle=True)
    #
    # mat = scipy.io.loadmat('data/MNISTtest.mat')
    # # test_data = torch.Tensor(mat['x']), torch.Tensor(mat['y']).squeeze().long().to(device)
    # test_loader2 = create_batches(x=torch.Tensor(mat['x']),
    #                               y=torch.Tensor(mat['y']).squeeze().long(),
    #                               batch_size=batch_size)

    model = model.to(device)
    #criteria1 = nn.CrossEntropyLoss()
    criteria2 = nn.BCEWithLogitsLoss()

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.90, weight_decay=1E-5)
    stopper = patienceStopper(epochs=epochs, patience=patience, printerval=printerval)

    print('Starting training...')

    def class2binary(x, c):
        b = torch.zeros_like(x)
        b[range(len(x)), c] = 1.0
        return b

    def train(model):
        for i, (x, y) in tqdm(enumerate(train_loader2), desc='Training', total=len(train_loader2)):
            x, y = x.to(device), y.to(device)
            # x = x.repeat([1, 3, 1, 1])  # grey to rgb
            # x /= 255.  # rescale to 0-1

            pred = model(x)
            loss = criteria2(pred, class2binary(pred, y))

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

    def test(model):
        # x, y = test_data
        for i, (x, y) in tqdm(enumerate(test_loader2), desc='Testing', total=len(test_loader2)):
            x, y = x.to(device), y.to(device)
            # x = x.repeat([1, 3, 1, 1])  # grey to rgb
            # x /= 255.  # rescale to 0-1

            yhat = model(x)
            loss = criteria2(yhat, class2binary(yhat, y))
            yhat_number = torch.argmax(yhat.data, 1)

            accuracy = []
            for j in range(nc):
                j = y == j
                accuracy.append((yhat_number[j] == y[j]).float().mean() * 100.0)

            return loss, accuracy

    for epoch in range(epochs):
        train(model.train())
        loss, accuracy = test(model.eval())
        if stopper.step(loss, metrics=(*accuracy,), model=model):
            break


if __name__ == '__main__':
    # model=MLP()
    # model = ConvNeta()
    # model = ConvNetb()

    # load pretrained model: https://github.com/Cadene/pretrained-models.pytorch#torchvision
    import pretrainedmodels

    model_name = 'resnet101'
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')

    # adjust last layer
    n = 2  # desired classes
    filters = model.last_linear.weight.shape[1]
    model.last_linear.bias = torch.nn.Parameter(torch.zeros(n))
    model.last_linear.weight = torch.nn.Parameter(torch.zeros(n, filters))
    model.last_linear.out_features = n

    # Display
    torch_utils.model_info(model, report='full')
    for x in ['model.input_size', 'model.input_space', 'model.input_range', 'model.mean', 'model.std']:
        print(x + ' =', eval(x))

    # Train
    main(model)
