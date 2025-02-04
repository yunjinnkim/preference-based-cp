import numpy as np
import torch

from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import train_test_split

from torch import nn

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import resnet18
from torch.optim import Adam


class LabelRankingResnet(nn.Module):
    """An image classifier based on a pretrained resnet18."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LabelRankingResnet, self).__init__()
        self.model = resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features

        self.model.fc = nn.Linear(num_ftrs, 10)

        self.effective_epochs = 0
        self.gradient_updates = 0

    def forward(self, x):
        return self.model(x)

    def fit(
        self,
        pairset_loader,
        learning_rate=0.01,
        num_epochs=100,
        random_state=None,
        verbose=False,
    ):
        self.classes_ = 10
        device = "cuda"

        optimizer = Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for inputs_a, labels_a, inputs_b, labels_b in pairset_loader:
                inputs_a, labels_a, inputs_b, labels_b  = inputs_a.to(device), labels_a.to(device), inputs_b.to(device), labels_b.to(device)
                labels_a = labels_a.unsqueeze(-1)
                labels_b = labels_b.unsqueeze(-1)
                optimizer.zero_grad()
                # inputs = torch.cat([inputs_a, inputs_b], dim=0)
                # outputs = self.model(inputs)
                outputs_a = self.model(inputs_a)
                outputs_b = self.model(inputs_b)
                # outputs_a, outputs_b = torch.split(outputs, inputs_a.size(0), dim=0)

                outputs_for_labels_a = outputs_a.gather(dim=1, index=labels_a)
                outputs_for_labels_b = outputs_b.gather(dim=1, index=labels_b)

                # Negative log-likelihood of Bradley-Terry model. Skill values are modeled as exp of neural network output
                loss = (
                    torch.log(
                        torch.exp(outputs_for_labels_a)
                        + torch.exp(outputs_for_labels_b)
                    )
                    - outputs_for_labels_a
                )

                loss = loss.mean()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if verbose:
                    print(f"epoch: {epoch} loss {loss}")

    def predict_class_skills(self, X):
        check_is_fitted(self, ["classes_"])

        if not self.num_classes:
            raise NotFittedError()
        self.eval()
        with torch.no_grad():
            skills = self.forward(torch.tensor(X, dtype=torch.float32))
        return skills

    def predict(self, X):
        """sklearn style predict function

        :param X: Features
        :return: Predicted class label
        """
        skills = self.predict_class_skills(X)
        return np.argmax(skills, axis=1)

    def __sklearn_is_fitted__(self):
        return True
