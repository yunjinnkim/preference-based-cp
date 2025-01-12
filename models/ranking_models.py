import numpy as np
import torch

from sklearn.exceptions import NotFittedError

from torch import nn

from models.simple_model import EarlyStopping
from util.ranking_datasets import (
    DyadOneHotPairDataset,
    MCDyadOneHotPairDataset,
    LabelPairDataset,
)

import torch
from torch.utils.data import DataLoader, random_split

from util.ranking_datasets import create_all_dyads, create_dyads


class LabelRankingModel(nn.Module):
    """Simple neural network model for label ranking.

    :param nn: _description_
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LabelRankingModel, self).__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.hidden = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.effective_epochs = 0
        self.gradient_updates = 0

    def forward(self, x):
        x = self.input(x)
        x = self.sigmoid(x)
        x = self.hidden(x)
        return x

    def _fit(
        self,
        train_loader,
        val_loader=None,
        learning_rate=0.01,
        num_epochs=100,
        random_state=None,
        patience=5,
        delta=0.0,
    ):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=10
        )

        self.effective_epochs = 0
        self.gradient_updates = 0

        self.train_losses = []
        self.val_losses = []
        if val_loader:
            early_stopping = EarlyStopping(patience=patience, delta=delta)

        # Training loop
        for epoch in range(num_epochs):
            self.train()
            train_loss = 0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self(inputs)
                outputs_for_labels = outputs.gather(dim=2, index=labels)
                loss = (
                    torch.log(
                        torch.exp(outputs_for_labels[:, 0])
                        + torch.exp(outputs_for_labels[:, 1])
                    )
                    - outputs_for_labels[:, 0]
                )
                loss = loss.mean()
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                self.gradient_updates += 1
            train_loss /= len(train_loader)

            # Validation step
            
            if val_loader:
                self.eval()
                val_loss = 0
                with torch.no_grad():

                    for val_inputs, val_labels in val_loader:
                        val_outputs = self(val_inputs)
                        val_outputs_for_labels = val_outputs.gather(dim=2, index=val_labels)
                        v_loss = (
                            torch.log(
                                torch.exp(val_outputs_for_labels[:, 0])
                                + torch.exp(val_outputs_for_labels[:, 1])
                            )
                            - val_outputs_for_labels[:, 0]
                        ).mean()
                        val_loss += v_loss.item()
                val_loss /= len(val_loader)
                self.val_losses.append(val_loss)

            self.train_losses.append(train_loss)

            # Step the scheduler based on validation loss
            if val_loader:
                scheduler.step(val_loss)
            else:
                scheduler.step(train_loss)
            self.effective_epochs += 1
            if val_loader:
                early_stopping(val_loss)
                if early_stopping.early_stop:
                    print("Stopping training.")
                    break

    def fit(
        self,
        X,
        y,
        num_epochs=100,
        learning_rate=0.01,
        num_classes=None,
        batch_size=32,
        val_frac=0.2,
        random_state=None,
        use_cross_instance_dataset=False,
        num_pairs=-1,
        patience=5,
        delta=0,
    ):
        """sklearn style fit function. Given classification data X and y, this first creates a
        dataset with a dyad ranking reresentation and then fits the model.
        The class labels are being one hot encoded.
        CAUTION: Here, a batch has batch_size * (num_classes - 1) pairwise preferences, as
        each example is transferred into (num_classes - 1) comaprisons

        :param train_loader: _description_
        :param learning_rate: _description_, defaults to 0.01
        :param num_epochs: _description_, defaults to 1000
        """
        if not num_classes:
            num_classes = len(np.unique(y))

        self.num_classes = num_classes
        if use_cross_instance_dataset:
            raise NotImplementedError()
            dataset = MCDyadOneHotPairDataset(
                X,
                y,
                num_classes=self.num_classes,
                random_state=random_state,
                num_pairs=num_pairs,
            )
        else:
            dataset = LabelPairDataset(X, y, num_classes=self.num_classes)
        gen = torch.Generator(device="cpu").manual_seed(random_state)

        train_dataset, val_dataset = random_split(
            dataset, [1 - val_frac, val_frac], generator=gen
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        self._fit(
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            random_state=random_state,
            patience=patience,
            delta=delta,
        )

    def predict_class_skills(self, X):
        if not self.num_classes:
            raise NotFittedError()
        self.eval()
        with torch.no_grad():
            skills = self.forward(torch.tensor(X, dtype=torch.float32))
        return skills.detach().cpu().numpy()

    def predict(self, X):
        """sklearn style predict function

        :param X: Features
        :return: Predicted class label
        """
        skills = self.predict_class_skills(X)
        return np.argmax(skills, axis=1)


class DyadRankingModel(nn.Module):
    """Simple neural network model for dyad ranking. Training data is assumed

    :param nn: _description_
    """

    def __init__(self, input_dim, hidden_dim):
        super(DyadRankingModel, self).__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.hidden = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.effective_epochs = 0
        self.gradient_updates = 0

    def forward(self, x):
        x = self.input(x)
        x = self.sigmoid(x)
        x = self.hidden(x)
        return x

    def _fit(
        self,
        train_loader,
        val_loader,
        learning_rate=0.01,
        num_epochs=100,
        random_state=None,
        patience=5,
        delta=0.0,
    ):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=10
        )

        self.effective_epochs = 0
        self.gradient_updates = 0

        self.train_losses = []
        self.val_losses = []

        early_stopping = EarlyStopping(patience=patience, delta=delta)

        # Training loop
        for epoch in range(num_epochs):
            self.train()
            train_loss = 0
            for inputs in train_loader:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = (
                    torch.log(torch.exp(outputs[:, 0]) + torch.exp(outputs[:, 1]))
                    - outputs[:, 0]
                )
                loss = loss.mean()
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                self.gradient_updates += 1
            train_loss /= len(train_loader)

            # Validation step
            self.eval()
            val_loss = 0
            with torch.no_grad():
                for val_inputs in val_loader:
                    val_outputs = self(val_inputs)
                    v_loss = (
                        torch.log(
                            torch.exp(val_outputs[:, 0]) + torch.exp(val_outputs[:, 1])
                        )
                        - val_outputs[:, 0]
                    ).mean()
                    val_loss += v_loss.item()
            val_loss /= len(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Step the scheduler based on validation loss
            scheduler.step(val_loss)
            self.effective_epochs += 1
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Stopping training.")
                break

    def fit(
        self,
        X,
        y,
        num_epochs=100,
        learning_rate=0.01,
        num_classes=None,
        batch_size=32,
        val_frac=0.2,
        random_state=None,
        use_cross_instance_dataset=False,
        num_pairs=-1,
        patience=5,
        delta=0,
    ):
        """sklearn style fit function. Given classification data X and y, this first creates a
        dataset with a dyad ranking reresentation and then fits the model.
        The class labels are being one hot encoded.
        CAUTION: Here, a batch has batch_size * (num_classes - 1) pairwise preferences, as
        each example is transferred into (num_classes - 1) comaprisons

        :param train_loader: _description_
        :param learning_rate: _description_, defaults to 0.01
        :param num_epochs: _description_, defaults to 1000
        """
        if not num_classes:
            num_classes = len(np.unique(y))

        self.num_classes = num_classes
        if use_cross_instance_dataset:
            dyadic_dataset = MCDyadOneHotPairDataset(
                X,
                y,
                num_classes=self.num_classes,
                random_state=random_state,
                num_pairs=num_pairs,
            )
        else:
            dyadic_dataset = DyadOneHotPairDataset(X, y, num_classes=self.num_classes)
        gen = torch.Generator(device="cpu").manual_seed(random_state)

        train_dataset, val_dataset = random_split(
            dyadic_dataset, [1 - val_frac, val_frac], generator=gen
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        self._fit(
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            random_state=random_state,
            patience=patience,
            delta=delta,
        )

    def predict(self, X):
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
        return class_preds.detach().cpu().numpy()

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
        return class_skills.detach().cpu().numpy()
