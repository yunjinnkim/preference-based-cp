import torch
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader, Subset
torch.set_default_device("cuda")
# Pretrained model
model = resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 100)
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
ds_train = CIFAR100(root="./data/", train=True, download=True, transform=transform)
ds_test = CIFAR100(root="./data/", train=False, download=True, transform=transform)
model.eval()  # for evaluation
finetune_loader = DataLoader(ds_train, batch_size=32)
test_loader = DataLoader(ds_test, batch_size=32)
### Finetune model
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.train()
# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in finetune_loader:
        inputs, labels = inputs.to("cuda"), labels.to("cuda")
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(finetune_loader)}")
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
saved_models = {}
def save_model_with_tracking(model, arch, save_path):
    if arch not in saved_models:
        torch.save(model.state_dict(), save_path)
        saved_models[arch] = save_path
        print(f"Model for arch {arch} saved at {save_path}")
    else:
        print(
            f"Model for arch {arch} is already saved at {saved_models[arch]}, skipping save."
        )
# Example usage
arch = model.__class__.__name__
save_path = f"./finetuned_models/model_classification_cifar100_{model.__class__.__name__}.pth"
save_model_with_tracking(model, arch, save_path)