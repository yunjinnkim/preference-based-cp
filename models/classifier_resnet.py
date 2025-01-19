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

class ClassifierResnet(nn.Module):
    """An image classifier based on a pretrained resnet18."""

    def __init__(self):
        super(ClassifierResnet, self).__init__()
        self.model = resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features

        self.model.fc = nn.Linear(num_ftrs, 10)
        
        self.effective_epochs = 0
        self.gradient_updates = 0

    def forward(self, x):
        return self.model(x)

    def fit(
        self,
        train_loader,
        learning_rate=0.01,
        num_epochs=100,
        random_state=None,
        verbose=False
    ):
        self.classes_ = 10

        optimizer = Adam(self.model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in train_loader:

                optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            if verbose:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%')

   
    def predict_class_skills(self, X):
        check_is_fitted(self, ["classes_"])

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

    def __sklearn_is_fitted__(self):
        return True