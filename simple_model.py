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
