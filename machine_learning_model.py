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
from PIL import Image
from io import BytesIO
from torchvision.transforms import v2

# visualizing images


"""
load data x
"""
# ===== Load dataset =====

data = load_dataset("DeadCardassian/PM25Vision")

# can change this transform
transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToTensor(),
])

# need this to load the data bc data is located in weird location
def collate_fn(batch):
    imgs = [transform(Image.open(BytesIO(x["image"])).convert("RGB")) for x in batch]
    labels = [x["pm25"] for x in batch]   # pm25 AQI value
    return torch.stack(imgs), torch.tensor(labels, dtype=torch.float32)

train_loader = DataLoader(data["train"], batch_size=100, shuffle=True, collate_fn=collate_fn) # creates batches (might want to change batch back to 32 for train loop)
for x_batch, y_batch in train_loader:
    print(x_batch.shape)
    print(y_batch.shape)
    break

"""
Display 100 Visualizations
"""
# classificiation ?
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
    pic = pic.permute(1, 2, 0)
    plt1 = plt.subplot(10, 10, idx + 1)
    plt1.imshow(pic)
    plt1.set_title(labels.item[idx])
    plt1.axis('off')
plt.tight_layout()
plt.show()












"""
Data Splicing / SCaling
Train 70%
Validation 15%
Test 15%
"""
#Working Here: Data Splitting
#train_data= FinalDataset(inputs, outputs, transforms)







"""
Model + Dataloader Class
"""

# Model class



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