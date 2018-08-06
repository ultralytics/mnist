import copy
import time

import numpy as np
import pickle


def create_batches(x, y, batch_size=1000, shuffle=False):
    if shuffle:
        # shuffle_data(x, y)
        rng_state = np.random.get_state()
        np.random.shuffle(x)
        np.random.set_state(rng_state)
        np.random.shuffle(y)

    nb = x.shape[0] // batch_size  # number of batches
    remainder = x.shape[0] % batch_size  # number of remainder samples
    if remainder != 0:
        print('Warning: dataset size %g indivisible by batch size %g, %g extra.' % (x.shape[0], batch_size, remainder))
        x, y = x[:nb * batch_size], y[:nb * batch_size]  # trim off extra data

    if type(x).__module__ == np.__name__:  # is numpy
        x = x.reshape(nb, batch_size, *x.shape[1:])
        y = y.reshape(nb, batch_size, *y.shape[1:])
    else:  # is pytorch
        x = x.view(nb, batch_size, *x.shape[1:])
        y = y.view(nb, batch_size, *y.shape[1:])
    batches = [(x[i], y[i]) for i in range(nb)]
    return batches


def normalize(x, axis=None):  # normalize x mean and std by axis
    if axis is None:
        mu, sigma = x.mean(), x.std()
    elif axis == 0:
        mu, sigma = x.mean(0), x.std(0)
    elif axis == 1:
        mu, sigma = x.mean(1).reshape(x.shape[0], 1), x.std(1).reshape(x.shape[0], 1)
    return (x - mu) / sigma, mu, sigma


def shuffle_data(x, y):  # randomly shuffle x and y by same axis=0 indices. no need to return values, shuffled in place
    rng_state = np.random.get_state()
    np.random.shuffle(x)
    np.random.set_state(rng_state)
    np.random.shuffle(y)


def split_data(x, y, train=0.7, validate=0.15, test=0.15, shuffle=False):  # split training data
    n = x.shape[0]
    if shuffle:
        shuffle_data(x, y)
    i = round(n * train)  # train
    j = round(n * validate) + i  # validate
    k = round(n * test) + j  # test
    return x[:i], y[:i], x[i:j], y[i:j], x[j:k], y[j:k]  # xy train, xy validate, xy test


def stdpt(r, ys):  # MSE loss + standard deviation (pytorch)
    r = r.detach()
    loss = (r ** 2).mean().cpu().item()
    std = r.std(0).cpu().numpy() * ys
    return loss, std


def modelinfo(model):
    nparams = sum(x.numel() for x in model.parameters())
    ngradients = sum(x.numel() for x in model.parameters() if x.requires_grad)
    print('%4s %26s %9s %12s %.20s' % ('', 'name', 'gradient', 'parameters', 'shape'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%4g %26s %9s %12g %.20s' % (i, name, p.requires_grad, p.numel(), list(p.shape)))
    print('\n%g layers, %g parameters, %g gradients' % (i + 1, nparams, ngradients))


class patienceStopper(object):
    def __init__(self, patience=10, verbose=True, epochs=1000, printerval=10, spa_start=float('inf')):
        self.patience = patience
        self.verbose = verbose
        self.bestepoch = 0
        self.bestmodel = None
        self.epoch = -1
        self.epochs = epochs - 1  # max epochs
        self.reset()
        self.t0 = time.time()
        self.t = self.t0
        self.printerval = printerval
        self.spa_start_epoch = spa_start
        self.spamodel = None

    def reset(self):
        self.bestloss = float('inf')
        self.bestmetrics = None
        self.num_bad_epochs = 0

    def step(self, loss, metrics=None, model=None):
        loss = loss.item()
        self.num_bad_epochs += 1
        self.epoch += 1
        self.first(model) if self.epoch == 0 else None
        self.printepoch(self.epoch, loss, metrics) if self.epoch % self.printerval == 0 else None

        if loss < self.bestloss:
            self.bestloss = loss
            self.bestmetrics = metrics
            self.bestepoch = self.epoch
            self.num_bad_epochs = 0
            if model:
                if self.bestmodel:
                    self.bestmodel.load_state_dict(model.state_dict())  # faster than deepcopy
                else:
                    self.bestmodel = copy.deepcopy(model)

        wa = self.epoch - self.spa_start_epoch + 1  # weight a
        if wa > 0:
            if self.spamodel:
                a = self.spamodel.state_dict()
                b = model.state_dict()
                wb = 1
                for key in a:
                    a[key] = (a[key] * wa + b[key] * wb) / (wa + wb)
                self.spamodel.load_state_dict(a)
            else:
                self.spamodel = copy.deepcopy(model)

        if self.num_bad_epochs > self.patience:
            self.final('%g Patience exceeded at epoch %g.' % (self.patience, self.epoch))
            return True
        elif self.epoch >= self.epochs:
            self.final('WARNING: %g Patience not exceeded by epoch %g (train longer).' % (self.patience, self.epoch))
            return True
        else:
            return False

    def first(self, model):
        if model:
            modelinfo(model)
        s = ('epoch', 'time', 'loss', 'metric(s)')
        print('%12s' * len(s) % s)

    def printepoch(self, epoch, loss, metrics):
        s = (epoch, time.time() - self.t, loss)
        if metrics is not None:
            for i in range(len(metrics)):
                s += (metrics[i],)
        p = '%12.5g' * len(s) % s
        print(p)
        with open('results.txt', 'a') as file:
            file.write(p + '\n')
        self.t = time.time()

    def final(self, msg):
        dt = time.time() - self.t0
        print('%s\nFinished %g epochs in %.3fs (%.3f epochs/s). Best results:' % (
            msg, self.epochs + 1, dt, (self.epochs + 1) / dt))
        self.printepoch(self.bestepoch, self.bestloss, self.bestmetrics)
