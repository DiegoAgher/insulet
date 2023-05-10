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

def get_interactions(data):
    interactions = []
    for col in ['baz', 'fyt', 'lgh']:
      for col_2 in ['bar',  'xgt', 'qgg', 'lux',
                    'yyz', 'drt', 'gox', 'foo', 'boz', 'hrt', 'juu']:
        col_name = '{}_{}'.format(col, col_2)
        data[col_name] = data[col] * data[col_2]
        interactions.append(col_name)
    return data, interactions