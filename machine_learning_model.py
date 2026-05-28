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

"""
AUGMENTATION FOR DATA IMAGES!!
"""
# can change this transform (transforms images into numbers for computer to understand) (for test and train)
t_test = v2.Compose([
    v2.ToTensor(),
    v2.Resize((224, 224)), # size of the images, will change depending on window size though
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

## transform for training data
t_train = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomPerspective(distortion_scale = 0.25, p=0.5),
    v2.ToTensor(),
    v2.Resize((224, 224)),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# need this to load the data bc data is located in weird location
def collate_fn_test(batch):
    imgs = [t_test(Image.open(BytesIO(x["image"])).convert("RGB")) for x in batch]
    labels = [x["pm25"] for x in batch]   # pm25 AQI value
    return torch.stack(imgs), torch.tensor(labels, dtype=torch.float32)

def collate_fn_train(batch):
    imgs = [t_train(Image.open(BytesIO(x["image"])).convert("RGB")) for x in batch]
    labels = [x["pm25"] for x in batch]   # pm25 AQI value
    return torch.stack(imgs), torch.tensor(labels, dtype=torch.float32)

### Make different transforms for train and test/val
### Have a collate_fn_train for train that uses train_transform
### Have a collate_fn_test for test/val that uses test_transform
### When creating dataloaders, pass in the corresponding collate functions to the collate_fn parameter

# test transform need, resize, if used normalize in train, then need to do for testing (RECOMENDED), turn it to a tensor 

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
train_loader = DataLoader(data["train"], batch_size=100, shuffle=True, collate_fn=collate_fn_train) # creates batches (might want to change batch back to 32 for train loop)
test_loader = DataLoader(data["test"], batch_size=32, shuffle=True, collate_fn=collate_fn_test)
val_loader = DataLoader(data["val"], batch_size=32, shuffle=True, collate_fn=collate_fn_test)

# view the data in train_loader, match input pictures with output PM2.5 concentration
for train_in, train_out in train_loader:
    print(train_in.shape)
    print(train_out.shape)
    break

for val_in, val_out in val_loader:
    print(val_in.shape)
    print(val_out.shape)
    break

for test_in, test_out in test_loader:
    print(test_in.shape)
    print(test_out.shape)
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
        self.layer1 = nn.Conv2d(3, 40,  kernel_size= 3, stride=1 , padding=1) # inputs to layer 1
        self.layer2 = nn.Conv2d(40, 80,  kernel_size= 3, stride=1 , padding=1) # inputs to layer 2
        self.layer3 = nn.Conv2d(80, 100,  3, 1 , 1) #The output can be modified for loss testing
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride= 2)
        self.fc1 = nn.Linear(28* 28 * 48, 400)
        self.fc2 = nn.Linear(400,5)
        self.relu = nn.ReLU() # activation function

    def forward(self, inputs):
<<<<<<< HEAD
        partial = self.relu(self.layer1(partial))
        partial = self.pool(partial)
        partial = self.relu(self.layer2(partial))
        partial = self.pool(partial)
        partial = self.relu(self.layer3(partial))
        partial = self.pool(partial)

        partial = partial.flatten(start_dim=1)
        partial = self.relu(self.fc1(partial))
        output = self.fc2(partial)
        return output
=======
 

model = myNN()
model.train() # puts model in training mode

# DO WE NEED THIS? ------------------------- ANTHONY
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
loss_function = nn.MSELoss()
# loss = loss_function(pred, outputs)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # define optimizer


"""
Training Loop

epoch = 100
#counter to know how many batches = len(dataloader)
for i in range(epoch):
    model.train() # sets the model into training mode, allows weights to be changed
    ### Get inputs and outputs in batches using the training DataLoader
    training_loss = [] # loss is a list (per eopch)
    for train_in, train_out in train_loader:
        train_preds = model(train_in) # runs the class which gets updated each loop
        loss = loss_function(train_preds, train_out.unsqueeze(1)) # loss would be a tensor
        #print(f"Epoch {epoch} | training loss: {loss.item()}")
        loss.backward() # calculates the slopes 
        optimizer.step() # updates weights aka .parameters
        optimizer.zero_grad() # removes all calculation don
        # add loss to list per batch
        training_loss.append(loss.item())
    # calculate RMSE for training (per epoch)
    train_RMSE = ((sum(training_loss))/(len(training_loss))) ** 0.5
    print(f"Epoch {i+1} | Training loss: {train_RMSE}")
    
        
    ### Gets input and outputs in batches using validation DataLoader
            # DO NOT TRAIN
    val_loss = []
    model.eval() # evaluation mode, even if i try to update weights, dont do it
    with torch.no_grad(): # tells pytorch to not calculate gradients, will run faster
        for val_in, val_out in val_loader:
            val_preds = model(val_in)
            loss = loss_function(val_preds, val_out.unsqueeze(1))
            #print(f"Epoch {epoch} | validation loss: {loss.item()}")
            # add loss to list per batch
            val_loss.append(loss.item())
    # calculate RMSE for training (per epoch)
    val_RMSE = ((sum(val_loss))/(len(val_loss))) ** 0.5
    print(f"Epoch {i+1} | Validation loss: {val_RMSE}")

"""

"""
Testing Loop
"""

model.eval() # weights cant be changed
with torch.no_grad(): # stops background running pytorch because we don't need it to calc slope anymore will made code faster
    ### Get inputs and outputs in batches using the testing DataLoader
    test_loss = []
    for test_in, test_out in test_loader:
        test_preds = model(test_in)
        loss = loss_function(test_preds, test_out) # ask eric, 'NoneType' object has no attribute 'size'
        #print(f"Epoch {epoch} | test loss: {loss.item()}")
        # add loss to list per batch
        test_loss.append(loss.item())
    # calculate RMSE for training (per epoch)
    test_RMSE = ((sum(test_loss))/(len(test_loss))) ** 0.5
    print(f"Testing loss: {test_RMSE}")

"""
Class Identifier + Output we not do this right?
"""