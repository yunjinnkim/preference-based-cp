import numpy as np
import torch

from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split

from torch import nn

import inspect

import random

from itertools import product

import torch
from torch.utils.data import Dataset, DataLoader, random_split


class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.verbose:
                    print(
                        f"Early stopping triggered. Best validation loss: {self.best_loss:.4f}"
                    )
                self.early_stop = True
