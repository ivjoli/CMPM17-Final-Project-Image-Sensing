"""
imports here
"""
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix

# needed to load the dataset
from datasets import load_dataset, DatasetDict
from PIL import Image
from io import BytesIO
from torchvision.transforms import v2

# visualizing images


"""
load data x
"""
# ===== Load dataset =====

data = load_dataset("DeadCardassian/PM25Vision") # loads the data from the website
#print(data) shows the data
#exit()

# can change this transform (transforms images into numbers for computer to understand)
transform = v2.Compose([
    v2.Resize((224, 224)), # size of the images, will change depending on window size though
    v2.ToTensor(),
])

# need this to load the data bc data is located in weird location
def collate_fn(batch):
    imgs = [transform(Image.open(BytesIO(x["image"])).convert("RGB")) for x in batch]
    labels = [x["pm25"] for x in batch]   # pm25 AQI value
    return torch.stack(imgs), torch.tensor(labels, dtype=torch.float32)


"""
Data Splicing / SCaling
Train 70%
Validation 15%
Test 15%
"""
# data already split into train and test
# need to grab validation from train
# splitting the test dataset into 50% test and 50% validation
dataset_split = data["test"].train_test_split(test_size=0.5) # grabs test dataset and splits it into 2 equal datasets
# create a new identification for data 
data = DatasetDict(
    {
        "train": data["train"],
        "val": dataset_split["train"],
        "test": dataset_split["test"],
    }
)

print(data)

"""
Batching Inputs
"""
# Batch train inputs/outputs
# Batch validation inputs/outputs
# batch test inputs/outputs
train_loader = DataLoader(data["train"], batch_size=100, shuffle=True, collate_fn=collate_fn) # creates batches (might want to change batch back to 32 for train loop)
test_loader = DataLoader(data["test"], batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(data["val"], batch_size=32, shuffle=True, collate_fn=collate_fn)

# view the data in train_loader, match input pictures with output PM2.5 concentration
for x_batch, y_batch in train_loader:
    print(x_batch.shape)
    print(y_batch.shape)
    break

"""
Display 100 Visualizations
"""
# classificiation ? can be put at the end for later
def map_pm25_to_class(pm25):
    if pm25 <= 50.4: return 0
    elif pm25 <= 100.4: return 1
    elif pm25 <= 150.4: return 2
    elif pm25 <= 200.4: return 3
    elif pm25 <= 300.4: return 4
    else: return 5

# create batches of images
# pics = batch of images
# pic = 1 image
for pics, labels in train_loader:
    break

#labels = [tensor(93), tensor(161), tensor(32)....]
#function that maps tensor value to AQI score: tensor(21) -> 0, tensor(250) -> 4
#apply that function to your list of labels to get a new list: [1, 3, 0....]
#these would be your labels you pass in to .set_title()

plt.figure(figsize = (20, 20)) # window size

# loop and display images
for idx, pic in enumerate(pics): 
    pic = pic.permute(1, 2, 0) # changes the order of dimentions to (W,H,C)
    plt1 = plt.subplot(10, 10, idx + 1) # size of image and indexing to the next image doing the same thing
    plt1.imshow(pic) # shows the image
    plt1.set_title(labels[idx]) # shows the labels
    plt1.axis('off')
plt.tight_layout()
plt.show()


"""
Model + Dataloader Class
"""
# Model class
class myNN(nn.Module):
    def __init__(self):
        super().__init__() # this calls a pytorch function to do math so no need to indent
        self.layer1 = nn.Linear(100, 40) # inputs to layer 1
        self.layer2 = nn.Linear(40,32) # inputs to layer 2
        self.layer9 = nn.Linear(30, 1)
        self.activation = nn.ReLU() # activation function

    def forward(self, inputs):
        partial = self.layer1(inputs)
        partial = self.activation(partial)
        partial = self.layer2(partial)
        partial = self.activation(partial)
        output = self.layer9(partial)
        return output

model = myNN()
model.train() # puts model in training mode


# Dataloader Class
class FinalDataset(Dataset):
    def __init__(self, input, transforms):
        self.transforms = transforms

# ivy put this here, can be changed or erased if Anthony deems so, not currently in use
class MyDataset(Dataset):
    def __init__(self, inputs, outputs):
        ### Initialize inputs and outputs
        self.inputs = inputs
        self.outputs = outputs
        self.length = len(inputs)

    def __len__(self):
        ### return the number of datapoints
        return self.length
    
    def __getitem__(self, idx):
        output = self.outputs[idx]
        input = self.inputs[idx]
        ### Get the input and output at index idx
        return input, output    




"""
Optimizer + Loss Function
"""
# optimizer = Adam with learning rate 0.01
# Loss Function = MSE


"""
Training Loop
"""
# Epoch == ??

"""
Testing Loop
"""

"""
Class Identifier + Output
"""