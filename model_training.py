import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset , DataLoader
import os

from Dataset import FaceKeypointDataset
from model import FaceKeypointModel
from train_steps import train
from utils import save_model

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 10
ROOT = 'data/new_data'

train_data = pd.read_csv(os.path.join(ROOT,'training.csv'))
valid_data = pd.read_csv(os.path.join(ROOT,'test.csv'))

train_dataset = FaceKeypointDataset(f'{ROOT}/training',train_data,True)
valid_dataset = FaceKeypointDataset(f'{ROOT}/test',valid_data,False)

train_dataloader = DataLoader(train_dataset,batch_size = 32,shuffle = True)
valid_dataloader = DataLoader(valid_dataset,batch_size = 32,shuffle = False)

print(f"Training sample instances: {len(train_data)}")
print(f"Validation sample instances: {len(valid_data)}")

model = FaceKeypointModel().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss = nn.SmoothL1Loss()

result = train(model = model,
    epochs = EPOCHS,
    train_dataloader= train_dataloader,
    valid_dataloader= valid_dataloader,
    optimizer = optimizer , 
    loss = loss ,
    device = DEVICE
               )

save_model(epochs = EPOCHS ,
           model = model ,
           optimizer = optimizer ,
           criterion= loss)






