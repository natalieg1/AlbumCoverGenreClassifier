import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision.models import alexnet, AlexNet_Weights

# Create the model
class AlexNetFreezeLast(nn.Module):
    def __init__(self, num_classes):
        super(AlexNetFreezeLast, self).__init__()
        self.alexnet = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        self.alexnet.classifier[6] = nn.Linear(4096, num_classes)

        for param in self.alexnet.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.alexnet(x)

# Define transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to match input size of AlexNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to match input size of AlexNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create the datasets
train_dataset = ImageFolder(root='album_covers/GAID/train/', transform=train_transform)
val_dataset = ImageFolder(root='album_covers/GAID/val/', transform=val_transform)
test_dataset = ImageFolder(root='album_covers/GAID/test/', transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlexNetFreezeLast(num_classes=len(train_dataset.classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

top_k_values = [1, 3, 5]
val_topk_accuracies = {k: [] for k in top_k_values}

num_epochs = 10
train_losses, val_losses = [], []
all_preds, all_labels = [], []

# Train the model
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    train_loss = running_loss / len(train_dataset)
    train_losses.append(train_loss)

    model.eval()
    running_loss = 0.0
    total = 0
    correct_counts = {k: 0 for k in top_k_values}
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            total += labels.size(0)
            _, topk_preds = outputs.topk(max(top_k_values), dim=1, largest=True, sorted=True)
            correct_mask = topk_preds.eq(labels.view(-1, 1).expand_as(topk_preds))
            for k in top_k_values:
                correct_counts[k] += correct_mask[:, :k].sum().item()

            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())  
            all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / len(val_dataset)
    val_losses.append(val_loss)

    for k in top_k_values:
        topk_accuracy = 100.0 * correct_counts[k] / total
        val_topk_accuracies[k].append(topk_accuracy)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Top-1 Accuracy: {val_topk_accuracies[1][-1]:.2f}%, Top-3 Accuracy: {val_topk_accuracies[3][-1]:.2f}%, Top-5 Accuracy: {val_topk_accuracies[5][-1]:.2f}%')

# Print out the plots
plt.figure(figsize=(12, 10))
plt.subplot(2, 1, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 1, 2)
for k in top_k_values:
    plt.plot(val_topk_accuracies[k], label=f'Top-{k} Accuracy')
plt.title('Top-k Validation Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()

cm = confusion_matrix(all_labels, all_preds)
cmd = ConfusionMatrixDisplay(cm, display_labels=train_dataset.classes)
fig, ax = plt.subplots(figsize=(12, 12))
cmd.plot(ax=ax)
plt.title('Validation Confusion Matrix')
plt.xticks(rotation=90)
plt.show()

# run the test data and print out the plots
model.eval()
test_total = 0
test_correct_counts = {k: 0 for k in top_k_values}
test_preds, test_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, topk_preds = outputs.topk(max(top_k_values), dim=1, largest=True, sorted=True)
        correct_mask = topk_preds.eq(labels.view(-1, 1).expand_as(topk_preds))
        for k in top_k_values:
            test_correct_counts[k] += correct_mask[:, :k].sum().item()

        test_preds.extend(outputs.argmax(dim=1).cpu().numpy())  # Only take the top-1 prediction
        test_labels.extend(labels.cpu().numpy())
        test_total += labels.size(0)

test_topk_accuracies = {k: 100.0 * test_correct_counts[k] / test_total for k in top_k_values}
print(f'Test Top-1 Accuracy: {test_topk_accuracies[1]:.2f}%, Test Top-3 Accuracy: {test_topk_accuracies[3]:.2f}%, Test Top-5 Accuracy: {test_topk_accuracies[5]:.2f}%')

cm = confusion_matrix(test_labels, test_preds)
cmd = ConfusionMatrixDisplay(cm, display_labels=test_dataset.classes)
fig, ax = plt.subplots(figsize=(12, 12))
cmd.plot(ax=ax)
plt.title('Test Confusion Matrix')
plt.xticks(rotation=90)
plt.show()
