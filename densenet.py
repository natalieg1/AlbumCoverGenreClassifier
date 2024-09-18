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
from torchvision.models import densenet201, DenseNet201_Weights

# Create the model
class DenseNetUnfreezeLastThreeWithDropout(nn.Module):
    def __init__(self, num_classes, dropout_p=0.8):
        super(DenseNetUnfreezeLastThreeWithDropout, self).__init__()
        self.densenet = densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1)
        self.densenet.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(self.densenet.classifier.in_features, num_classes)
        )

        # Freeze all layers and then unfreeze the last three dense blocks
        for param in self.densenet.parameters():
            param.requires_grad = False
        
        for param in self.densenet.features.denseblock3.parameters():
            param.requires_grad = True
        for param in self.densenet.features.transition3.parameters():
            param.requires_grad = True
        for param in self.densenet.features.denseblock4.parameters():
            param.requires_grad = True
        for param in self.densenet.features.norm5.parameters():
            param.requires_grad = True
        for param in self.densenet.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.densenet(x)

# Transforms and data augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to match input size of DenseNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets with data augmentation for training
train_dataset = ImageFolder(root='album_covers/GAID/train/', transform=train_transform)
val_dataset = ImageFolder(root='album_covers/GAID/val/', transform=val_transform)
test_dataset = ImageFolder(root='album_covers/GAID/test/', transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# set up model for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DenseNetUnfreezeLastThreeWithDropout(num_classes=len(train_dataset.classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0005, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

top_k_values = [1, 3, 5]
val_topk_accuracies = {k: [] for k in top_k_values}

num_epochs = 20
train_losses, val_losses = [], []
all_preds, all_labels = [], []

# train the model
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
    scheduler.step(val_loss)

    for k in top_k_values:
        topk_accuracy = 100.0 * correct_counts[k] / total
        val_topk_accuracies[k].append(topk_accuracy)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Top-1 Accuracy: {val_topk_accuracies[1][-1]:.2f}%, Top-3 Accuracy: {val_topk_accuracies[3][-1]:.2f}%, Top-5 Accuracy: {val_topk_accuracies[5][-1]:.2f}%')

# plot the training results
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
plt.title('Confusion Matrix')
plt.xticks(rotation=90)
plt.show()

# Run and print the test set
model.eval()
test_total = 0
test_correct_counts = {k: 0 for k in top_k_values}
test_preds = []  # Initialize as an empty list
test_labels = []  # Initialize as an empty list

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
