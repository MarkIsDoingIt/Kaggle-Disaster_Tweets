from torch.utils.data import Dataset
import os
import pandas as pd
import torch
import json
import pdb


class TextDatset(Dataset):
    def __init__(self, split, model_use):
        file_name = split + '.csv'
        self.file_path = os.path.join('/home/broca/kg/data', file_name)
        self.data = pd.read_csv(self.file_path)
        self.split = split
        self.model_use = model_use
    
    def __getitem__(self, idx):
        id = self.data['id'][idx]
        text = self.data['text'][idx]
        if self.split == 'test':
            target = -1
            if self.model_use == 't5':
                target = str(target)
            return id, text, target
        else:
            target = self.data['target'][idx]
            if self.model_use == 't5':
                target = str(target)
            return id, text, target
    
    def __len__(self):
        return len(self.data)
    
