import pickle
import torch
import numpy as np
import pandas as pd

import torchvision.transforms as transforms
import torchvision.datasets as datasets

transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize the image to 224x224
    transforms.ToTensor(),         # Convert the image to a PyTorch tensor
    transforms.Normalize(         # Normalize the image with mean and standard deviation
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# open the file for reading binary data
with open('xgb_feature_names.pkl', 'rb') as f:
    # read the list from the file using pickle.load()
    xgb_feature_names = pickle.load(f)
