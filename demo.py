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

# confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# import model class
from machine_learning_model import myNN

model = myNN()
model.load_state_dict(model.load_state_dict(torch.load("model_version_1.pt", weights_only=True)))

pic = Image.open("demo_pic.webp").convert("RGB")

transform = v2.Compose([
    v2.ToTensor(),
    v2.Resize((224, 224)), # size of the images, will change depending on window size though
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

pic = transform(pic)

# classificiation ? can be put at the end for later
def map_pm25_to_class(pm25):
    if pm25 <= 50.4: return 0
    elif pm25 <= 100.4: return 1
    elif pm25 <= 150.4: return 2
    elif pm25 <= 200.4: return 3
    elif pm25 <= 300.4: return 4
    else: return 5

"""""
Using
"""
# takes in image
# pass image to model

loss_function = nn.MSELoss()

model.eval() # weights cant be changed
with torch.no_grad(): # stops background running pytorch because we don't need it to calc slope anymore will made code faster
    ### Get inputs and outputs in batches using the testing DataLoader
    test_loss = []
    AQI_values_in = [] # stores prediction
    AQI_values_out = [] # stores labels

    test_preds = model(pic)
    loss = loss_function(test_preds, "IMAGE ANSWER") # ask eric, 'NoneType' object has no attribute 'size'
    #print(f"Epoch {epoch} | test loss: {loss.item()}")
    # add loss to list per batch
    test_loss.append(loss.item())
    AQI_values_in.extend(test_preds.tolist())
    AQI_values_out.extend("IMAGE ANSWER".tolist())
    
    # calculate RMSE for training (per epoch)
    test_RMSE = ((sum(test_loss))/(len(test_loss))) ** 0.5
    print(f"Testing loss: {test_RMSE}")
    # passing in AQI values into classification function for class level
    AQI_class_level = [] # predictions
    for pm25 in AQI_values_in:
        AQI_class_level.append(map_pm25_to_class(pm25[0]))

    AQI_class_out = [] # actual
    for pm25 in AQI_values_out:
        AQI_class_out.append(map_pm25_to_class(pm25))



