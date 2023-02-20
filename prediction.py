
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from einops.layers.torch import Rearrange, Reduce

img_size = 224
batch_size = 4 
# Get a list of all image files in the data directory

# TODO: save and load model, only train if model file isn't there
# TODO: report accuracy

class CancerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
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
    
model = SegmentationModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
losses = []
for epoch in range(1, 10):
    for step_id, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if step_id % 10 == 0:
            avg_losses = np.array(losses)
            print(f"Epoch:{epoch}\tStep:{step_id+1}\tCrossEntropyLoss:{avg_losses.mean():0.2f} ± {avg_losses.std():0.2f}")
            losses = []

print('Finished Training')
test_image = Image.open("tumour.jpg")

test_image = transform(test_image).to(device) # make sure to unsqueeze the tensor to add a batch dimension
with torch.no_grad():
    output = model(test_image.unsqueeze(0))

prediction = torch.argmax(output, dim=1).item() # get the index of the predicted class
if prediction == 0:
    print("Non-cancerous")
else:
    print("Cancerous")