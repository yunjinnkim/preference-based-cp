import torch
import numpy as np
from torch import nn

import torch
from torch.utils.data import Dataset, DataLoader


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

    def __init__(self, X, y, num_classes=None):
        if not num_classes:
            num_classes = max(y) + 1
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


class ClassifierModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ClassifierModel, self).__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.hidden = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.input(x)
        x = self.relu(x)
        x = self.hidden(x)
        return x

    def fit(self, train_loader, learning_rate=0.01, num_epochs=100):
        loss_fn = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

        # optimization loop
        for epoch in range(num_epochs):
            self.train()

            for i, (inputs, labels) in enumerate(train_loader):
                print(labels)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

    def predict_proba(self, X):
        self.eval()
        with torch.no_grad():
            logits = self.forward(torch.tensor(X, dtype=torch.float32))
            probabilities = torch.softmax(logits, dim=1)
        return probabilities.numpy()

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)


class DyadRankingModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(DyadRankingModel, self).__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.num_classes = num_classes

        # output dimension is always 1
        self.hidden = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input(x)
        x = self.sigmoid(x)
        x = self.hidden(x)
        return x

    def fit(self, train_loader, learning_rate=0.001, num_epochs=1000):
        loss_fn = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

        # optimization loop
        for epoch in range(num_epochs):
            self.train()

            for i, inputs in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = self(inputs)

                loss = (
                    torch.log(torch.exp(outputs[:, 0]) + torch.exp(outputs[:, 1]))
                    - outputs[:, 0]
                )

                # loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

    # def predict_proba(self, X):
    #     self.eval()
    #     with torch.no_grad():
    #         logits = self.forward(torch.tensor(X, dtype=torch.float32))
    #         probabilities = torch.softmax(logits, dim=1)
    #     return probabilities.numpy()

    # def predict_class_label(self, X):
    #     # create dyads
    #     eye = torch.eye(self.num_classes)
    #     eye = eye.repeat(X.)
    #     x_long = x.
    #     x_long.

    #     probabilities = self.predict_proba(X)
    #     return np.argmax(probabilities, axis=1)
