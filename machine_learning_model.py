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
from datasets import load_dataset
import torchvision.transforms as T #maybe from torchvision.utils.data import Transforms
from PIL import Image
from io import BytesIO

"""
load data + data cleaning 
"""
# ===== Load dataset =====

data = load_dataset("DeadCardassian/PM25Vision")

# can change this transform
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])

# need this to load the data bc data is located in weird location
def collate_fn(batch):
    imgs = [transform(Image.open(BytesIO(x["image"])).convert("RGB")) for x in batch]
    labels = [x["pm25"] for x in batch]   # pm25 AQI value
    return torch.stack(imgs), torch.tensor(labels, dtype=torch.float32)

train_loader = DataLoader(data["train"], batch_size=32, shuffle=True, collate_fn=collate_fn)
# df = data["train"].to_pandas()
# print(df)
# print(df.head())

train_loader = DataLoader(data["train"], batch_size=32, shuffle=True, collate_fn=collate_fn)

"""
Data Splicing / SCaling
Train 70%
Validation 15%
Test 15%
"""
#Working Here: Data Splitting
train_data= FinalDataset(inputs, outputs, transforms)
train_loader = DataLoader(train_data, batch_size=32)
for x_batch, y_batch in train_loader:
    print(x_batch.shape())
    print(y_batch.shape())
    break








"""
Model + Dataloader Class
"""

# Model class




transforms = Transforms.compose([
    transforms.ToTensor()
    transforms.Rotate(-30,30)
    transforms.Hflip(p= 0.5)
])

# Dataloader Class
class FinalDataset(Dataset):
    def __init__(self, input, transforms):
        self.transforms = transforms

"""
Batching Inputs
"""

# Batch train inputs/outputs
# Batch validation inputs/outputs
# batch test inputs/outputs



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