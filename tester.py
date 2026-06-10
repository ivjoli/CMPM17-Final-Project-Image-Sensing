import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau as ReduceLROnPlateau
import torch 
import wandb
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

# confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def map_pm25_to_class(pm25):
    if pm25 <= 50.4: return 0
    elif pm25 <= 100.4: return 1
    elif pm25 <= 150.4: return 2
    elif pm25 <= 200.4: return 3
    elif pm25 <= 300.4: return 4
    else: return 5

t_test = v2.Compose([
      v2.Resize((224, 224)),
      v2.ToTensor(),
      v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

class myNN(nn.Module):
    def __init__(self):
        super().__init__() # this calls a pytorch function to do math so no need to indent
        
        self.layer1 = nn.Conv2d(3, 40,  kernel_size= 3, stride=1 , padding=1) # inputs to layer 1
        self.batchNorm1 = nn.BatchNorm2d( 40 )        
        self.layer1b= nn.Conv2d(40, 40, 3, 1, 1)
        self.batchNorm1b = nn.BatchNorm2d( 40 )

        self.layer2 = nn.Conv2d(40, 80,  kernel_size= 3, stride=1 , padding=1) # inputs to layer 2
        self.batchNorm2 = nn.BatchNorm2d( 80 )
        self.layer2b= nn.Conv2d(80, 80, 3, 1, 1)
        self.batchNorm2b = nn.BatchNorm2d( 80 )

        self.layer3 = nn.Conv2d(80, 50,  kernel_size= 3, stride=1 , padding=1) # inputs to layer 2
        self.batchNorm3 = nn.BatchNorm2d( 50 )
        self.layer3b= nn.Conv2d(50, 50, 3, 1, 1)
        self.batchNorm3b = nn.BatchNorm2d( 50 )

        self.layer4 = nn.Conv2d(50, 100,  3, 1 , 1) #The output can be modified for loss testing
        self.batchNorm4 = nn.BatchNorm2d( 100 )
        self.layer4b= nn.Conv2d(100, 100, 3, 1, 1)
        self.batchNorm4b = nn.BatchNorm2d( 100 )

        self.pool = nn.MaxPool2d(kernel_size=2, stride= 2)
        self.fc1 = nn.Linear(14 * 14 * 100, 400)
        self.fc2 = nn.Linear(400,1)
        self.relu = nn.ReLU() # activation function
    #Forward Pass for Milestone 2!
    def forward(self, inputs):

        partial = self.relu(self.batchNorm1(self.layer1(inputs)))
        partial = self.relu(self.batchNorm1b(self.layer1b(partial)))
        partial = self.pool(partial)

        partial = self.relu(self.batchNorm2(self.layer2(partial)))
        partial = self.relu(self.batchNorm2b(self.layer2b(partial)))
        partial = self.pool(partial)

        partial = self.relu(self.batchNorm3(self.layer3(partial)))
        partial = self.relu(self.batchNorm3b(self.layer3b(partial)))
        partial = self.pool(partial)

        partial = self.relu(self.batchNorm4(self.layer4(partial)))
        partial = self.relu(self.batchNorm4b(self.layer4b(partial)))
        partial = self.pool(partial)

        partial = partial.flatten(start_dim=1)
        partial = self.relu(self.fc1(partial))
        output = self.fc2(partial)
        return output

device = "cuda" if torch.cuda.is_available() else "cpu"

model = myNN()
model.load_state_dict(torch.load("pm25_best_AQI", map_location=device,weights_only=True))
model = model.to(device)
model.eval()
imagePath= "demo_pic.jpg"
img = Image.open(imagePath).convert("RGB")
img= t_test(img)

with torch.no_grad():
    pred_scaled = model(img.unsqueeze(0).to(device)).item()
    pred_pm25 = pred_scaled * 100
    pred_class = map_pm25_to_class(pred_pm25)
print(f"Based off Image-- Predicted PM2.5 is {pred_pm25}")
print(f"Based off Image-- Predicted AQI class is {pred_class}")