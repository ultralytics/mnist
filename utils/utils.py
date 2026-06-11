# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import copy
import random
import time

import numpy as np
import torch

from . import torch_utils

# Set printoptions
torch.set_printoptions(linewidth=320, precision=8)
np.set_printoptions(linewidth=320, formatter={"float_kind": "{:11.5g}".format})  # format short g, %precision=5


def init_seeds(seed=0):
    """Initialize random seeds for reproducibility across various libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch_utils.init_seeds(seed=seed)


def create_batches(x, y, batch_size=1000, shuffle=False):
    """Generate batches from input data `x` and `y` with optional shuffle and specified batch size."""
    if shuffle:
        # shuffle_data(x, y)
        rng_state = np.random.get_state()
        np.random.shuffle(x)
        np.random.set_state(rng_state)
        np.random.shuffle(y)

    nb = x.shape[0] // batch_size  # number of batches
    remainder = x.shape[0] % batch_size  # number of remainder samples
    if remainder != 0:
        print(f"Warning: dataset size {x.shape[0]:g} indivisible by batch size {batch_size:g}, {remainder:g} extra.")
        x, y = x[: nb * batch_size], y[: nb * batch_size]  # trim off extra data

    if type(x).__module__ == np.__name__:  # is numpy
        x = x.reshape(nb, batch_size, *x.shape[1:])
        y = y.reshape(nb, batch_size, *y.shape[1:])
    else:  # is pytorch
        x = x.view(nb, batch_size, *x.shape[1:])
        y = y.view(nb, batch_size, *y.shape[1:])
    return [(x[i], y[i]) for i in range(nb)]


def normalize(x, axis=None):  # normalize x mean and std by axis
    """Normalize input array x along the specified axis, returning normalized array, mean, and standard deviation."""
    if axis is None:
        mu, sigma = x.mean(), x.std()
    elif axis == 0:
        mu, sigma = x.mean(0), x.std(0)
    elif axis == 1:
        mu, sigma = x.mean(1).reshape(x.shape[0], 1), x.std(1).reshape(x.shape[0], 1)
    return (x - mu) / sigma, mu, sigma


def shuffle_data(x, y):  # randomly shuffle x and y by same axis=0 indices. no need to return values, shuffled in place
    """Randomly shuffle the arrays x and y in place along the same axis=0 indices."""
    rng_state = np.random.get_state()
    np.random.shuffle(x)
    np.random.set_state(rng_state)
    np.random.shuffle(y)


def split_data(x, y, train=0.7, validate=0.15, test=0.15, shuffle=False):  # split training data
    """Splits arrays x and y into training, validation, and test sets with specified ratios, optionally shuffling
    them.
    """
    n = x.shape[0]
    if shuffle:
        shuffle_data(x, y)
    i = round(n * train)  # train
    j = round(n * validate) + i  # validate
    k = round(n * test) + j  # test
    return x[:i], y[:i], x[i:j], y[i:j], x[j:k], y[j:k]  # xy train, xy validate, xy test


class patienceStopper:
    """Implements early stopping mechanism to halt training when performance stops improving, preventing overfitting."""

    def __init__(self, patience=10, verbose=True, epochs=1000, printerval=10, spa_start=float("inf")):
        """Initialize patienceStopper with given parameters for controlling early stopping in model training."""
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
        """Resets training state variables including best loss, best metrics, and number of bad epochs."""
        self.bestloss = float("inf")
        self.bestmetrics = None
        self.num_bad_epochs = 0

    def step(self, loss, metrics=None, model=None):
        """Updates training state variables and logs progress for each epoch, optionally handling metrics and model
        state.
        """
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
            self.final(f"{self.patience:g} Patience exceeded at epoch {self.epoch:g}.")
            return True
        elif self.epoch >= self.epochs:
            self.final(f"WARNING: {self.patience:g} Patience not exceeded by epoch {self.epoch:g} (train longer).")
            return True
        else:
            return False

    def first(self, model):
        """Logs model information and prints the header for training metrics."""
        if model:
            torch_utils.model_info(model)
        s = ("epoch", "time", "loss", "metric(s)")
        print("%12s" * len(s) % s)

    def printepoch(self, epoch, loss, metrics):
        """Prints and logs epoch number, elapsed time, loss, and metrics during training."""
        s = (epoch, time.time() - self.t, loss)
        if metrics is not None:
            for i in range(len(metrics)):
                s += (metrics[i],)
        p = "%12.5g" * len(s) % s
        print(p)
        with open("results.txt", "a") as file:
            file.write(p + "\n")
        self.t = time.time()

    def final(self, msg):
        """Prints and logs the final training summary including total epochs, time taken, and best metrics."""
        dt = time.time() - self.t0
        print(
            f"{msg}\nFinished {self.epochs + 1:g} epochs in {dt:.3f}s ({(self.epochs + 1) / dt:.3f} epochs/s). Best results:"
        )
        self.printepoch(self.bestepoch, self.bestloss, self.bestmetrics)
