import torch
import numpy as np

from sklearn.exceptions import NotFittedError

from torch import nn

import torch
from torch.utils.data import Dataset, DataLoader, random_split


class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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


class ClassifierModel(nn.Module):
    """Simple neural network for classification"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ClassifierModel, self).__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.hidden = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input(x)
        x = self.sigmoid(x)
        x = self.hidden(x)
        return x

    def _fit(self, train_loader, val_loader, learning_rate=0.001, num_epochs=100):
        """Torch implementation for fitting the neural network

        :param train_loader: Loader for training data
        :param learning_rate: Learning rate for the optimizer, defaults to 0.001
        :param num_epochs: Number of epochs, defaults to 100
        """

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=10
        )

        loss_fn = nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(num_epochs):
            self.train()
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

            # Validation step
            self.eval()
            val_loss = 0
            with torch.no_grad():
                for val_inputs, val_labels in val_loader:
                    val_outputs = self(val_inputs)
                    val_loss += loss_fn(val_outputs, val_labels).item()
            val_loss /= len(val_loader)

            # Step the scheduler based on validation loss
            scheduler.step(val_loss)

        # optimization loop
        for epoch in range(num_epochs):
            self.train()

            for i, (inputs, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

    def fit(
        self, X, y, learning_rate=0.001, num_epochs=100, batch_size=32, val_frac=0.2
    ):
        """sklearn style function that takes X and y in order to fit the neural network

        :param X: _description_
        :param y: _description_
        :param learning_rate: _description_, defaults to 0.01
        :param num_epochs: _description_, defaults to 100
        :param batch_size: _description_, defaults to 32
        """
        dataset = TabularDataset(X, y)
        train_dataset, val_dataset = random_split(dataset, [1 - val_frac, val_frac])
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        self._fit(
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
        )

    def predict_proba(self, X):
        """sklearn style predict_proba function

        :param X: Features
        :return: Predicted probability distribution
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(torch.tensor(X, dtype=torch.float32))
            probabilities = torch.softmax(logits, dim=1)
        return probabilities.numpy()

    def predict(self, X):
        """sklearn style predict function

        :param X: Features
        :return: Predicted class label
        """
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)


class DyadRankingModel(nn.Module):
    """Simple neural network model for dyad ranking. Training data is assumed

    :param nn: _description_
    """

    def __init__(self, input_dim, hidden_dim):
        super(DyadRankingModel, self).__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.hidden = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input(x)
        x = self.sigmoid(x)
        x = self.hidden(x)
        return x

    def _fit(self, train_loader, val_loader, learning_rate=0.001, num_epochs=100):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=10
        )

        # Training loop
        for epoch in range(num_epochs):
            self.train()
            for inputs in train_loader:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = (
                    torch.log(torch.exp(outputs[:, 0]) + torch.exp(outputs[:, 1]))
                    - outputs[:, 0]
                )
                loss = loss.mean()
                loss.backward()
                optimizer.step()

            # Validation step
            self.eval()
            val_loss = 0
            with torch.no_grad():
                for val_inputs in val_loader:
                    val_outputs = self(val_inputs)
                    loss = (
                        torch.log(torch.exp(outputs[:, 0]) + torch.exp(outputs[:, 1]))
                        - outputs[:, 0]
                    ).mean()
                    val_loss += loss.item()
            val_loss /= len(val_loader)

            # Step the scheduler based on validation loss
            scheduler.step(val_loss)

    def fit(
        self,
        X,
        y,
        num_epochs=100,
        learning_rate=0.001,
        num_classes=None,
        batch_size=32,
        val_frac=0.2,
    ):
        """sklearn style fit function. Given classification data X and y, this first creates a
        dataset with a dyad ranking reresentation and then fits the model.
        The class labels are being one hot encoded.
        CAUTION: Here, a batch has batch_size * (num_classes - 1) pairwise preferences, as
        each example is transferred into (num_classes - 1) comaprisons

        :param train_loader: _description_
        :param learning_rate: _description_, defaults to 0.001
        :param num_epochs: _description_, defaults to 1000
        """
        if not num_classes:
            num_classes = y.max() + 1

        self.num_classes = num_classes
        dyadic_dataset = DyadOneHotPairDataset(X, y, num_classes=self.num_classes)
        train_dataset, val_dataset = random_split(
            dyadic_dataset, [1 - val_frac, val_frac]
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        self._fit(
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
        )

    def predict_class(self, X):
        """Convenience function that allows to use the ranker as an sklearn style classifier

        :param X: Features
        """
        if not self.num_classes:
            raise NotFittedError()
        self.eval()
        dyads = create_all_dyads(X, self.num_classes)
        with torch.no_grad():
            preds = self(dyads)
            class_preds = preds.view(-1, self.num_classes).argmax(axis=1)
        return class_preds.detach().numpy()

    def predict_class_skills(self, X):
        """Convenience function that allows to use the ranker as an sklearn style classifier

        :param X: Features
        """
        if not self.num_classes:
            raise NotFittedError()
        self.eval()
        dyads = create_all_dyads(X, self.num_classes)
        with torch.no_grad():
            skills = self(dyads)
            class_skills = skills.view(-1, self.num_classes)
        return class_skills.detach().numpy()
