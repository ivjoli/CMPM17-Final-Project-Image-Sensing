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
import torchvision.transforms as T
from PIL import Image
from io import BytesIO

"""
load data + data cleaning 
"""
# ===== Load dataset =====
ds = load_dataset("DeadCardassian/PM25Vision")



"""
Data Splicing / SCaling
Train 70%
Validation 15%
Test 15%
"""







"""
Model + Dataloader Class
"""

# Model class





# Dataloader Class


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