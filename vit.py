import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import vit_b_16
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Create the model
class ViTModel(nn.Module):
    def __init__(self, num_classes):
        super(ViTModel, self).__init__()
        self.model = vit_b_16(pretrained=True)
        
        # Freeze everything then unfreeze the last two layers
        for param in self.model.parameters():
            param.requires_grad = False
        
        for param in self.model.encoder.layers[-2].parameters():
            param.requires_grad = True  
        for param in self.model.encoder.layers[-1].parameters():
            param.requires_grad = True 
        
        for param in self.model.heads.head.parameters():
            param.requires_grad = True

        # Match the number of classes and add dropout
        num_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Sequential(
            nn.Dropout(0.7), 
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Data transforms and data augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),  
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root=root, transform=transform, target_transform=target_transform)

    def find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir() and d.name != '.ipynb_checkpoints']
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

# Load the data
train_dataset = ImageFolder(root='album_covers/GAID/train/', transform=train_transform)
val_dataset = ImageFolder(root='album_covers/GAID/val/', transform=val_transform)
test_dataset = ImageFolder(root='album_covers/GAID/test/', transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# set up model for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViTModel(num_classes=len(train_dataset.classes)).to(device)
criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)
scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = 20
train_losses, val_losses = [], []
val_topk_accuracies = {k: [] for k in [1, 3, 5]}  

# Run the training 
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
    correct_counts = {k: 0 for k in [1, 3, 5]}
    all_preds, all_labels = [], []  

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            total += labels.size(0)

            _, topk_preds = outputs.topk(5, dim=1, largest=True, sorted=True)
            correct_mask = topk_preds == labels.view(-1, 1)

            for k in [1, 3, 5]:
                correct_counts[k] += correct_mask[:, :k].any(dim=1).float().sum().item()

            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / len(val_dataset)
    val_losses.append(val_loss)
    scheduler.step(val_loss)

    for k in [1, 3, 5]:
        topk_accuracy = 100 * correct_counts[k] / total
        val_topk_accuracies[k].append(topk_accuracy)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Top-k Accuracies: {[f"Top-{k}: {val_topk_accuracies[k][-1]:.2f}%" for k in [1, 3, 5]]}')

# Plot training loss and val accuracy
plt.figure(figsize=(12, 10))
plt.subplot(2, 1, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 1, 2)
for k in [1, 3, 5]:
    plt.plot(val_topk_accuracies[k], label=f'Top-{k} Accuracy')
plt.title('Top-k Validation Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()

# Run test set and print results
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
model.eval()
test_loss = 0.0
total = 0
correct_counts = {k: 0 for k in [1, 3, 5]}
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * images.size(0)
        total += labels.size(0)

        _, topk_preds = outputs.topk(5, dim=1, largest=True, sorted=True)
        correct_mask = topk_preds == labels.view(-1, 1)

        for k in [1, 3, 5]:
            correct_counts[k] += correct_mask[:, :k].any(dim=1).float().sum().item()

        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_loss = test_loss / len(test_dataset)
test_topk_accuracies = {k: 100 * correct_counts[k] / total for k in [1, 3, 5]}

print(f'Test Loss: {test_loss:.4f}')
print(f'Test Top-1 Accuracy: {test_topk_accuracies[1]:.2f}%')
print(f'Test Top-3 Accuracy: {test_topk_accuracies[3]:.2f}%')
print(f'Test Top-5 Accuracy: {test_topk_accuracies[5]:.2f}%')


# Plot the confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=test_dataset.classes)
disp.plot(xticks_rotation='vertical')
plt.show()
