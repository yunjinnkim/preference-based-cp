# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from math import ceil
from sklearn.ensemble import RandomForestClassifier
import openml

from torch.utils.data import Dataset
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import Subset

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# %%
def visualize_pair(img1, class1, img2, class2, class_names, prob1=None, prob2=None):
    """
    Visualize a comparison pair, including the images, their labels, and probabilities.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    # Image 1
    axes[0].imshow(img1.permute(1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
    if prob1:
        axes[0].set_title(f"Class {class1} ({class_names[class1]})\nProb: {prob1:.2f}")
    else:
        axes[0].set_title(f"Class {class1} ({class_names[class1]})")
    axes[0].axis("off")

    # Image 2
    axes[1].imshow(img2.permute(1, 2, 0))
    if prob2:
        axes[1].set_title(f"Class {class2} ({class_names[class2]})\nProb: {prob2:.2f}")
    else:
        axes[1].set_title(f"Class {class2} ({class_names[class2]})")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


# %%
def get_rowwise_pairs(probs, i_threshold=None, j_threshold=None):
    result = []
    for row_idx, row in enumerate(probs):
        i_indices, j_indices = np.where(row[:, None] > row)
        if i_threshold is not None:
            mask_i = row[i_indices] > i_threshold
            i_indices, j_indices = i_indices[mask_i], j_indices[mask_i]
        if j_threshold is not None:
            mask_j = row[j_indices] > j_threshold
            i_indices, j_indices = i_indices[mask_j], j_indices[mask_j]
        pairs = np.column_stack(
            (np.full(i_indices.shape, row_idx), i_indices, j_indices)
        )
        result.append(pairs)
    try:
        result = np.vstack(result)
    except:
        result = np.array([])
    return result


def get_rowwise_pairs_with_max(matrix, j_threshold=None):
    result = []
    for row_idx, row in enumerate(matrix):
        max_value = np.max(row)
        max_ids = np.argwhere(row == max_value)
        for max_idx in max_ids:
            j_indices = np.where(row < max_value)[0]
            if j_threshold is not None:
                j_indices = j_indices[row[j_indices] > j_threshold]

            pairs = np.column_stack(
                (
                    np.full(j_indices.shape, row_idx),
                    np.full(j_indices.shape, row_idx),
                    np.full(j_indices.shape, max_idx),
                    j_indices,
                )
            )
        result.append(pairs)
    try:
        result = np.vstack(result)
    except:
        result = np.array([])
    return result


def get_cross_row_pairs(matrix):
    num_rows, num_cols = matrix.shape
    result = []
    # Iterate over all pairs of rows (k, l)
    for k in range(num_rows):
        for l in range(num_rows):
            if k != l:
                # Compare all pairs of elements from row k and row l
                i_indices, j_indices = np.where(matrix[k][:, None] > matrix[l])
                # Combine row indices (k, l) with column indices (i, j)
                pairs = np.column_stack(
                    (
                        np.full(i_indices.shape, k),
                        i_indices,
                        np.full(j_indices.shape, l),
                        j_indices,
                    )
                )
                result.append(pairs)
    try:
        result = np.stack(result)
    except:
        result = np.array([])
    return result


def get_cross_row_pairs_with_max(matrix):
    """Generates pairs between argmax classes across instances

    :param matrix: _description_
    :return: _description_
    """
    result = []

    max_indices = [np.where(row == row.max())[0] for row in matrix]

    pairs = []

    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if i != j:
                # Compare all combinations of maxima indices between row i and row j
                for col_i in max_indices[i]:
                    for col_j in max_indices[j]:
                        if matrix[i, col_i] > matrix[j, col_j]:
                            pairs.append((i, j, col_i, col_j))
    try:
        result = np.stack(np.array(pairs))
    except:
        result = np.array([])

    return result


# %%


class PairwiseCIFAR10H(Dataset):

    def sample_rows(array, sample):
        if isinstance(sample, float):  # Fraction of rows
            num_rows = int(sample * array.shape[0])
        elif isinstance(sample, int):  # Number of rows
            num_rows = sample
        else:
            raise ValueError("Sample must be a float (fraction) or int (number).")

        sampled_indices = np.random.choice(array.shape[0], size=num_rows, replace=False)
        return array[sampled_indices]

    def __init__(self, dataset, probs, in_instance_pairs=1.0, cross_instance_pairs=1.0):

        self.dataset = dataset
        self.probs = probs
        print("Generating in-instance pairs:")
        in_instance_pairs = get_rowwise_pairs_with_max(self.probs)
        print("Generating cross-instance pairs:")
        cross_instance_pairs = get_cross_row_pairs_with_max(self.probs)
        self.pair_indices = np.vstack([in_instance_pairs, cross_instance_pairs])

    def __len__(self):
        return len(self.pair_indices)

    def __getitem__(self, idx):
        img_a_idx, img_b_idx, label_a, label_b = self.pair_indices[idx]
        img_a, ground_truth_a = self.dataset[img_a_idx]
        img_b, ground_truth_b = self.dataset[img_b_idx]
        return img_a, label_a, img_b, label_b


# %%
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # ResNet expects 224x224 input
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)
dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

class_names = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
probs = np.load("data/cifar10h-probs.npy")

subset = Subset(dataset, range(0, 5))
subset_probs = probs[0:5]

pair_data = PairwiseCIFAR10H(subset, subset_probs)

# %%

# Set device (use GPU if available)

# Step 1: Load the CIFAR-10 dataset
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # ResNet expects 224x224 input
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
    ]
)

batch_size = 64

# Training and test data loaders
trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
subset = Subset(testset, indices=range(0, 5))

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
subset_loader = torch.utils.data.DataLoader(
    dataset=subset, batch_size=batch_size, shuffle=True
)

# %%
model_clf = models.resnet18(pretrained=True)
num_ftrs = model_clf.fc.in_features

model_clf.fc = nn.Linear(num_ftrs, 10)
model_clf = model_clf.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_clf.parameters(), lr=0.001)

epochs = 5
for epoch in range(epochs):
    model_clf.train()
    running_loss = 0.0
    for inputs, labels in subset_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model_clf(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(trainloader):.4f}")


# %%
model_clf.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in subset_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model_clf(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")


# %%
pairset = PairwiseCIFAR10H(subset, probs[:5])
batch_size_scale = len(pairset) / len(subset)
pairset_loader = torch.utils.data.DataLoader(
    dataset=pairset, batch_size=ceil(batch_size), shuffle=True
)

model_rnk = models.resnet18(pretrained=True)
num_ftrs = model_rnk.fc.in_features

model_rnk.fc = nn.Linear(num_ftrs, 10)
model_rnk = model_rnk.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_rnk.parameters(), lr=0.001)

epochs = 5
for epoch in range(epochs):
    model_rnk.train()
    running_loss = 0.0
    for inputs_a, labels_a, inputs_b, labels_b in pairset_loader:
        inputs_a, labels_a, inputs_b, labels_b = (
            inputs_a.to(device),
            labels_a.to(device),
            inputs_b.to(device),
            labels_b.to(device),
        )
        labels_a = labels_a.unsqueeze(-1)
        labels_b = labels_b.unsqueeze(-1)
        optimizer.zero_grad()
        # inputs = torch.cat([inputs_a, inputs_b], dim=0)
        # outputs = model_rnk(inputs)
        outputs_a = model_rnk(inputs_a)
        outputs_b = model_rnk(inputs_b)
        # outputs_a, outputs_b = torch.split(outputs, inputs_a.size(0), dim=0)

        outputs_for_labels_a = outputs_a.gather(dim=1, index=labels_a)
        outputs_for_labels_b = outputs_b.gather(dim=1, index=labels_b)

        # Negative log-likelihood of Bradley-Terry model. Skill values are modeled as exp of neural network output
        loss = (
            torch.log(torch.exp(outputs_for_labels_a) + torch.exp(outputs_for_labels_b))
            - outputs_for_labels_a
        )
        loss = loss.mean()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(trainloader):.4f}")


# %%

model_rnk.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in subset_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model_rnk(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
