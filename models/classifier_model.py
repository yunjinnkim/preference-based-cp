import numpy as np
import torch

from torch import nn

from models.simple_model import EarlyStopping

from util.ranking_datasets import TabularDataset


import torch
from torch.utils.data import DataLoader, random_split
from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.utils.validation import check_is_fitted


class ClassifierModel(nn.Module, ClassifierMixin, BaseEstimator):
    """Simple neural network for classification"""

    def __init__(self, input_dim, hidden_dims, output_dim, activations=None):
        super(ClassifierModel, self).__init__()

        layers = []
        in_features = input_dim

        if activations:
            for hidden_size, activation in zip(hidden_dims, activations):
                layers.append(nn.Linear(in_features, hidden_size))
                layers.append(activation)  # Add activation function
                in_features = hidden_size
        else:

            for hidden_size in hidden_dims:
                layers.append(nn.Linear(in_features, hidden_size))
                layers.append(nn.Sigmoid())  # Add activation function
                in_features = hidden_size

        # Add the final output layer
        layers.append(nn.Linear(in_features, output_dim))

        # Combine all layers into a single nn.Sequential module
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def _fit(
        self,
        train_loader,
        val_loader,
        learning_rate=0.01,
        num_epochs=100,
        patience=5,
        delta=0.0,
    ):
        """Torch implementation for fitting the neural network

        :param train_loader: Loader for training data
        :param learning_rate: Learning rate for the optimizer, defaults to 0.01
        :param num_epochs: Number of epochs, defaults to 100
        """

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=10
        )

        loss_fn = nn.CrossEntropyLoss()
        early_stopping = EarlyStopping(patience=patience, delta=delta)

        self.effective_epochs = 0
        self.gradient_updates = 0

        self.train_losses = []
        self.val_losses = []

        # Training loop
        for epoch in range(num_epochs):
            train_loss = 0
            self.train()
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = loss_fn(outputs, labels)
                train_loss += loss_fn(outputs, labels).item()
                loss.backward()
                optimizer.step()
                self.gradient_updates += 1
            train_loss /= len(train_loader)
            # Validation step
            self.eval()
            val_loss = 0

            # Validation step
            if val_loader is not None:
                self.eval()
                val_loss = 0
                with torch.no_grad():
                    for val_inputs, val_labels in val_loader:
                        val_outputs = self(val_inputs)
                        val_loss += loss_fn(val_outputs, val_labels).item()
                val_loss /= len(val_loader)
                self.val_losses.append(val_loss)

            self.train_losses.append(train_loss)

            # Step the scheduler based on validation loss
            if val_loader:
                scheduler.step(val_loss)
                self.effective_epochs += 1
                early_stopping(val_loss)
                if early_stopping.early_stop:
                    print("Stopping training.")
                    break
            else:
                scheduler.step(train_loss)
            print(train_loss)

    def fit(
        self,
        X,
        y,
        learning_rate=0.01,
        num_epochs=100,
        batch_size=32,
        val_frac=0.2,
        random_state=0,
        patience=5,
        delta=0,
    ):
        """sklearn style function that takes X and y in order to fit the neural network

        :param X: _description_
        :param y: _description_
        :param learning_rate: _description_, defaults to 0.01
        :param num_epochs: _description_, defaults to 100
        :param batch_size: _description_, defaults to 32
        """
        self.classes_ = np.unique(y)
        dataset = TabularDataset(X, y)
        gen = torch.Generator(device="cuda").manual_seed(random_state)
        val_loader = None
        if val_frac > 0.0:
            train_dataset, val_dataset = random_split(
                dataset, [1 - val_frac, val_frac], generator=gen
            )
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset

            train_loader = DataLoader(train_dataset, batch_size=batch_size)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        else:
            train_dataset = dataset
            train_loader = DataLoader(train_dataset, batch_size=batch_size)
        self._fit(
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            patience=patience,
            delta=delta,
        )
        self.fitted = True

    def predict_proba(self, X):
        """sklearn style predict_proba function

        :param X: Features
        :return: Predicted probability distribution
        """
        check_is_fitted(self, ["classes_"])

        self.eval()
        with torch.no_grad():
            logits = self.forward(torch.tensor(X, dtype=torch.float32))
            probabilities = torch.softmax(logits, dim=1)
        return probabilities.detach().cpu().numpy()

    def predict(self, X):
        """sklearn style predict function

        :param X: Features
        :return: Predicted class label
        """
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

    def __sklearn_is_fitted__(self):
        return self.fitted

    def classes_(self):
        return
