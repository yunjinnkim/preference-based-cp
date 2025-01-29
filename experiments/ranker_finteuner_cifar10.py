import torch
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import torchvision.models as models


torch.set_default_device("cuda")
# Pretrained model
model = resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 10)
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
ds_train = CIFAR10(root="./data/", train=True, download=True, transform=transform)
ds_test = CIFAR10(root="./data/", train=False, download=True, transform=transform)
# ds_train = Subset(ds_train,list(range(500)))
# ds_test = Subset(ds_test,list(range(500)))
model.eval()  # for evaluation
test_loader = DataLoader(ds_test, batch_size=32)
### Finetune model
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.train()

# Random dataset that always returns new images
class RandomPairDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.length = len(base_dataset)
        self.device="cuda"
        self.generator = torch.Generator(device=self.device).manual_seed(torch.initial_seed())
    def __len__(self):
        return self.length  # Can be any arbitrary number

    def __getitem__(self, index):
        # Sample two random indices
        idx1, idx2 = torch.randint(0, self.length, (2,))
        
        # Get the corresponding image-label pairs
        img1, label1 = self.base_dataset[idx1]

        all_labels = torch.arange(10, device="cuda")  # [0, 1, ..., 9]
        possible_wrong_labels = all_labels[all_labels != label1]  # Exclude true label

        # Randomly select one wrong label
        label2 = possible_wrong_labels[torch.randint(0, len(possible_wrong_labels), (1,), generator=self.generator)].item()

        return img1, label1, img1, label2  

randomized_dataset = RandomPairDataset(ds_train)
generator = torch.Generator(device="cuda").manual_seed(42)  # Or use torch.initial_seed()
finetune_loader = DataLoader(randomized_dataset, batch_size=32, shuffle=True, generator=generator)

# Load Pretrained ResNet

# Training Loop
save_every_n_batches = 250  # Save model every N batches
total_iterations = 50000  # Total training steps (since epochs don't apply)

for iteration in range(total_iterations):
    for (images1, labels1, images2, labels2) in finetune_loader:
        if iteration >= total_iterations:
            break  # Stop training after total_iterations

        images1, labels1, images2, labels2 = images1.to("cuda"), labels1.to("cuda"),images2.to("cuda"), labels2.to("cuda")
        
        optimizer.zero_grad()
        outputs1 = model(images1)
        outputs2 = model(images2)
        outputs_for_labels1 = outputs1.gather(dim=1, index=labels1.unsqueeze(-1))
        outputs_for_labels2 = outputs2.gather(dim=1, index=labels2.unsqueeze(-1))
        loss = (
            torch.log(
                torch.exp(outputs_for_labels1)
                + torch.exp(outputs_for_labels2)
            )
            - outputs_for_labels1
        )
        loss = loss.mean()

        loss.backward()
        optimizer.step()
    
    # Save model checkpoint periodically
    if iteration % save_every_n_batches == 0:
        torch.save(model.state_dict(), f"./finetuned_models/resnet_ranker_cifar10_iter{iteration}.pth")
        print(f"Saved model at iteration {iteration}")

    # Print progress
    if iteration % 100 == 0:
        print(f"Iteration {iteration}/{total_iterations} - Loss: {loss.item():.4f}")

print("Training complete.")
model.eval()
correct = 0
total = 0
with torch.no_grad():  # Disable gradient computation
    for images, labels in test_loader:
        images, labels = images.to("cuda"), labels.to("cuda")
        # Forward pass
        outputs = model(images)
        # Get predicted class (max logit value)
        _, predicted = torch.max(outputs, 1)
        # Compare predictions with true labels
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
# Calculate accuracy
accuracy = correct / total
print(accuracy)