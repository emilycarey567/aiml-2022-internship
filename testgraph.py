import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from einops.layers.torch import Rearrange, Reduce
import pathlib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import f1_score
from matplotlib.ticker import MaxNLocator

img_size = 224
batch_size = 4 
# Get a list of all image files in the data directory

class CancerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.tumor = 0
        self.healthy = 0
        for sub_dir in os.listdir(root_dir):
            if sub_dir.startswith('.'):
                continue
            if sub_dir == "cancerous":
                label = 1
            else:
                label = 0
                
            sub_path = os.path.join(root_dir, sub_dir)
            for filename in os.listdir(sub_path):
                if filename.startswith('.'):
                    continue
                
                if label == 1:
                    self.tumor += 1
                elif label == 0:
                    self.healthy += 1

                self.images.append(os.path.join(sub_path, filename))
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = np.array(Image.open(img_path).convert('L'))
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((img_size, img_size)),
])

train_dataset = CancerDataset("train_data", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

class SegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
        )
        self.global_avgpool = Reduce('b c h w -> b c', 'mean')
        self.classifier = nn.Linear(in_features=128, out_features=2)

    def forward(self, x):
        features = self.features(x)
        # print(f"{features.shape=}")
        per_feature_response = self.global_avgpool(features)
        # print(f"{per_feature_response.shape=}")
        logits = self.classifier(per_feature_response)
        return logits
    
MODEL_PATH = 'medical_model.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not pathlib.Path(MODEL_PATH).is_file():
    model = SegmentationModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    train_losses=[]
    train_accuracies = []
    train_f1scores = []
    for epoch in range(0, 30):
        epoch_losses = []
        epoch_accuracies = []
        epoch_f1scores=[]
        
        for step_id, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            _, predicted = torch.max(outputs.data, axis=1)
            epoch_accuracies.append((predicted==labels).sum() / len(labels))
            epoch_f1scores.append(f1_score(labels, predicted.tolist(), average='binary') )

            print(f"{epoch=} {step_id=} {loss.item()=:0.3f}")

        epoch_losses = np.array(epoch_losses) 
        train_losses.append(epoch_losses.mean())

        epoch_accuracies=np.array(epoch_accuracies)
        train_accuracies.append(epoch_accuracies.mean())

        epoch_f1scores=np.array(epoch_f1scores)
        train_f1scores.append(epoch_f1scores.mean())

    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train Loss')
    plt.savefig('train_loss.jpeg')
    plt.show()

    plt.plot(train_accuracies, label='Train Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train Accuracy')
    plt.savefig('train_Accuracy.jpeg')
    plt.show()

    plt.plot(train_f1scores, label='Train f1 score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('Train f1score')
    plt.savefig('train_f1scores.jpeg')
    plt.show()

    print('Finished Training')
    torch.save(model.state_dict(), MODEL_PATH)

model = SegmentationModel()
assert pathlib.Path(MODEL_PATH).is_file()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
print("loaded model")

# load test dataset
test_dataset = CancerDataset("test_data", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# evaluate on test dataset
criterion = nn.CrossEntropyLoss()
model.eval()

test_losses = []
test_predicted = []
test_labels = []
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_losses.append(loss.item())
        _, predicted = torch.max(outputs.data, axis=1)
        test_predicted.extend(predicted.cpu().tolist())
        test_labels.extend(labels.cpu().tolist())

avg_test_loss = np.array(test_losses).mean()
test_accuracy: float = (np.array(test_predicted) == np.array(test_labels)).sum() / len(test_labels)
test_f1_score: float = f1_score(test_labels, test_predicted, average='binary')


print(f"""
{avg_test_loss=:0.4f}
{test_accuracy*100=:0.2f}%
{test_f1_score*100=:0.2f}%

Number of tumor images in training set {sum((1 for label in train_dataset.labels if label == 1))}
Number of healthy images in training set {sum((1 for label in train_dataset.labels if label == 0))}

Number of tumor images in testing set {sum((1 for label in test_dataset.labels if label == 1))}
Number of healthy images in testing set {sum((1 for label in test_dataset.labels if label == 0))}

""")