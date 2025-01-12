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


def create_dyads(X, y, num_classes):
    X = torch.tensor(X, dtype=torch.float32)
    eye = torch.eye(num_classes)
    y_oh = eye[y]
    dyads = torch.cat([X, y_oh], dim=1)
    return dyads


def create_all_dyads(X, num_classes):
    X = torch.tensor(X, dtype=torch.float32)
    eye = torch.eye(num_classes)
    eyes = eye.repeat((X.shape[0], 1))
    Xs = X.repeat_interleave(num_classes, dim=0)
    dyads = torch.cat([Xs, eyes], dim=1)
    return dyads


class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DyadOneHotPairDataset(Dataset):
    """Given classificaiton data X,y, this generates a dataset of dyad pairs, where the class labels are one hot encoded and all possible preferences are present. The first alternative is the preferred one.

    :param Dataset: _description_
    """

    def __init__(self, X, y, num_classes):
        dyad_pairs = []
        eye = torch.eye(num_classes)

        for X_vec, label in zip(X, y):
            for k in range(0, num_classes):
                if k == label:
                    continue
                dyad_pairs.append(
                    torch.vstack(
                        [
                            torch.hstack(
                                [torch.tensor(X_vec, dtype=torch.float32), eye[label]]
                            ),
                            torch.hstack(
                                [torch.tensor(X_vec, dtype=torch.float32), eye[k]]
                            ),
                        ]
                    )
                )

        self.dyad_pairs = torch.stack(dyad_pairs)

    def __len__(self):
        return len(self.dyad_pairs)

    def __getitem__(self, idx):
        return self.dyad_pairs[idx]


class LabelPairDataset(Dataset):
    """Given classificaiton data X,y, this generates a dataset of label ranking pairs, where the class labels are encoded as indices and all possible preferences are present. The first alternative is the preferred one.

    :param Dataset: _description_
    """

    def __init__(self,X=None, y=None, num_classes=None):
        if X is not None:
            self.create_from_classification_data(X,y,num_classes)

    def create_from_numpy_pairs(self, X_pairs, y_pairs):
        self.X_pairs = torch.tensor(X_pairs,dtype=torch.float32)
        self.y_pairs = torch.tensor(y_pairs,dtype=torch.long)


    def create_from_classification_data(self,X, y, num_classes):
        X_pairs = []
        y_pairs = []

        for X_vec, label in zip(X, y):
            for k in range(0, num_classes):
                if k == label:
                    continue
                X_pairs.append(
                    torch.vstack(
                        [
                            torch.tensor(X_vec, dtype=torch.float32),
                            torch.tensor(X_vec, dtype=torch.float32),
                        ]
                    )
                )
                y_pairs.append(
                    torch.vstack(
                        [
                            torch.tensor(label, dtype=torch.long),
                            torch.tensor(k, dtype=torch.long),
                        ]
                    )
                )

        self.X_pairs = torch.stack(X_pairs)
        self.y_pairs = torch.stack(y_pairs)


    def __len__(self):
        return len(self.y_pairs)

    def __getitem__(self, idx):
        return self.X_pairs[idx], self.y_pairs[idx]


class MCPairDatasetFromSoftLabels(Dataset):
    """Given classificaiton data X,y, this generates a dataset of label ranking pairs, where the class labels are encoded as indices and all possible preferences are present. The first alternative is the preferred one.

    :param Dataset: _description_
    """

    def __init__(self, dataset, soft_labels, num_classes, cross_instance_pairs=None):
        """

        :param dataset: _description_
        :param soft_labels: _description_
        :param num_classes: _description_
        :param cross_instance_pairs: _description_, defaults to None
        """

        for i, (X_vec, label) in enumerate(dataset):
            for k in range(0, num_classes):
                if k == label:
                    continue
                X_pairs.append(
                    torch.vstack(
                        [
                            torch.tensor(X_vec, dtype=torch.float32),
                            torch.tensor(X_vec, dtype=torch.float32),
                        ]
                    )
                )
                y_pairs.append(
                    torch.vstack(
                        [
                            torch.tensor(label, dtype=torch.long),
                            torch.tensor(k, dtype=torch.long),
                        ]
                    )
                )

        self.X_pairs = torch.stack(X_pairs)
        self.y_pairs = torch.stack(y_pairs)

    def __len__(self):
        return len(self.y_pairs)

    def __getitem__(self, idx):
        return self.X_pairs[idx], self.y_pairs[idx]


class MCDyadOneHotPairDataset(Dataset):
    """Given classificaiton data X,y, this generates a dataset of dyad pairs, where the class labels are one hot encoded and all possible preferences are present. The first alternative is the preferred one.

    :param Dataset: _description_
    """

    def __init__(self, X, y, num_classes, random_state=0, num_pairs=-1):
        dyads_true = []
        dyads_false = []
        eye = torch.eye(num_classes)

        for X_vec, label in zip(X, y):
            dyads_true.append(
                torch.hstack([torch.tensor(X_vec, dtype=torch.float32), eye[label]])
            )
            for k in range(0, num_classes):
                if k != label:
                    dyads_false.append(
                        torch.hstack([torch.tensor(X_vec, dtype=torch.float32), eye[k]])
                    )

        indices_true = range(len(dyads_true))
        indices_false = range(len(dyads_false))
        indices = list(product(list(indices_true), list(indices_false)))
        random.Random(random_state).shuffle(indices)
        dyad_pairs = []
        for index in indices[:num_pairs]:
            dyad_pairs.append(
                torch.vstack([dyads_true[index[0]], dyads_false[index[1]]])
            )

        self.dyad_pairs = torch.stack(dyad_pairs)

    def __len__(self):
        return len(self.dyad_pairs)

    def __getitem__(self, idx):
        return self.dyad_pairs[idx]
