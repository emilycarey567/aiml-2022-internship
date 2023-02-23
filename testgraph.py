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

# TODO: report accuracy during training
# TODO: you need a testing set, so gather a test_dataset, and instead of just running inference on one image as you've done, load an image at a time and report average loss and average accuracy for the testing set.
# TODO: learn about https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
# TODO: report f1_score = sklearn.metrics.f1_score(labels, predicted_labels).
# TODO: report how many cancerous images are in your training and testing set, and how many non-canerous images are in your train/test set. 

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
    
MODEL_PATH = 'medical_model.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if pathlib.Path(MODEL_PATH).is_file():
    print("Doing something")
    model = SegmentationModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    train_losses=[]

    # plot_losses = []
    # losses = []
    # train_losses=[]
    # train_accs = []
    # train_f1s = []
    # av_loss_array=[]
    for epoch in range(1, 3):
        epoch_losses = []
        # epoch_correct = 0
        # epoch_f1 = 0.0
        # epoch_total = 0
        
        for step_id, data in enumerate(train_loader):
            if step_id==5:
                break
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            # output.shape == [B, Class]
            # find the max value in the class, and report the index. 0 or 1, and we know a 1 is a cancer img
            # using that, does class_idx == label_idx
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            # epoch_loss += loss.item() * inputs.size(0)
            # _, predicted = torch.max(outputs.data, 1)
            # epoch_correct += (predicted == labels).sum().item()
            # epoch_total += labels.size(0)
            # epoch_f1 += f1_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='binary') * labels.size(0)
            # print("Doing something")
            # if step_id % 10 == 0:
            #     avg_losses = np.array(losses)
            #     print(f"Epoch:{epoch}\tStep:{step_id+1}\tCrossEntropyLoss:{avg_losses.mean():0.2f} ± {avg_losses.std():0.2f}")
            #     plot_losses.append(avg_losses.mean())
            #     losses = 

            #     train_losses.append(epoch_loss / epoch_total)
            #     train_acc = 100 * epoch_correct / epoch_total
            #     train_f1 = epoch_f1 / epoch_total
            #     train_accs.append(train_acc)
            #     train_f1s.append(train_f1)
            #     plt.plot(train_losses, label='train loss')
 
            #     plt.legend()
            #     plt.show()

        epoch_losses = np.array(epoch_losses) 
        avg_epoch_losses = epoch_losses.mean()
        train_losses.append(avg_epoch_losses)

    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train Loss')
    plt.savefig('train_loss.jpeg')
    plt.show()

    print('Finished Training')
    # torch.save(model.state_dict(), MODEL_PATH)

model = SegmentationModel()
assert pathlib.Path(MODEL_PATH).is_file()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
print("loaded model")


# test_image = np.array(Image.open("tumour.jpg").convert('L'))

# test_image = transform(test_image).to(device)
# # output = model(test_image.unsqueeze(0))

# prediction = torch.argmax(output, dim=1).item() # get the index of the predicted class
# if prediction == 0:
    # print("Non-cancerous")
# else:
    # print("Cancerous")


# load test dataset
test_dataset = CancerDataset("test_data", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# evaluate on test dataset
criterion = nn.CrossEntropyLoss()
model.eval()
test_losses = []
test_predicted = []
test_labels = []
test_accs=[]
tests_f1=[]
tests_f1s=[]
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_losses.append(loss.item())
        _, predicted = torch.max(outputs.data, axis=1)
        test_predicted += predicted.cpu().numpy().tolist()
        test_labels += labels.cpu().numpy().tolist()
        test_accuracys = 100 * np.sum(np.array(test_predicted) == np.array(test_labels)) / len(test_labels)
        test_accs.append(test_accuracys)
        tests_f1 = f1_score(test_labels, test_predicted, average='binary')
        tests_f1s.append(f1_score(test_labels, test_predicted, average='binary'))
test_loss = np.mean(test_losses)
test_accuracy = 100 * np.sum(np.array(test_predicted) == np.array(test_labels)) / len(test_labels)
#test_accs.append(test_accuracy)
from sklearn.metrics import f1_score
test_f1 = f1_score(test_labels, test_predicted, average='binary')
#tests_f1.append(f1_score(test_labels, test_predicted, average='binary'))
print('Loss on test dataset: %0.2f' % (test_loss))
print('Accuracy on test dataset: %0.2f %%' % (test_accuracy))
print('F1 score on test dataset: %0.2f' % (test_f1))

# print number of cancerous images in train and test datasets
num_train_cancerous = sum(train_dataset.labels)
num_test_cancerous = sum(test_dataset.labels)
test_dataset = CancerDataset("test_data", transform=transform)

#num_non_cancerous_test = len(test_dataset) - num_cancerous_test
print('Number of cancerous images in train dataset: %d' % (num_train_cancerous))
print('Number of cancerous images in test dataset: %d' % (num_test_cancerous))

#plt.plot(train_losses, label='train loss')
plt.plot(test_losses, label='test loss')
plt.legend()
plt.show()

#plt.plot(train_accs, label='train accuracy')
plt.plot(test_accs, label='test accuracy')
plt.legend()
plt.show()

#plt.plot(train_f1s, label='train f1 score')
plt.plot(tests_f1s, label='test f1 score')
plt.legend()
plt.show()