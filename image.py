import pandas as pd
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter

train_data = pd.read_csv('train.csv')
val_data = pd.read_csv('val.csv')

class ImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        label = self.dataframe.iloc[idx, 1]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.Resize((128, 128)),  
    transforms.ToTensor(),          
    transforms.Normalize((0.5,), (0.5,))  
])

train_dataset = ImageDataset(dataframe=train_data, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

val_dataset = ImageDataset(dataframe=val_data, transform=transform)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1) 
        self.conv2 = nn.Conv2d(16, 32, 3, 1) 
        self.pool = nn.AdaptiveAvgPool2d((6, 6))

        self.fc1 = nn.Linear(32 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    running_loss = 0.0
    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    prediction_counts = Counter()

    with torch.no_grad():
        for images, labels in val_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            prediction_counts.update(predicted.cpu().numpy())

    print(f'Epoch {epoch+1}, Training Loss: {running_loss/len(train_dataloader)}, Validation Loss: {val_loss/len(val_dataloader)}, Accuracy: {100 * correct / total:.2f}%')
print(f"Prediction Distribution: {dict(prediction_counts)}")
