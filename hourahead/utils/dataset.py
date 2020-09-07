import os
import pandas as pd
import torch

from torch.utils.data.dataset import Dataset

class Dataset(object):
    
    def __init__(self, path, timesteps):
        self.data_path = path
        self.timesteps = timesteps
        self.train_data_raw = pd.read_excel(os.path.join(path, "train_in.xlsx"), header=None).values[:, :6]
        self.train_target_raw = pd.read_excel(os.path.join(path, "train_out.xlsx"), header=None).values
        self.test_data_raw = pd.read_excel(os.path.join(path, "test_in.xlsx"), header=None).values[:, :6]
        self.test_target_raw = pd.read_excel(os.path.join(path, "test_out.xlsx"), header=None).values
        self.train_data, self.train_target = self.process_train()
        self.test_data, self.test_target = self.process_test()
    
    def process_train(self):
        train_data, train_target = list(), list()

        for i in range(24, len(self.train_data_raw), 24):
            for j in range(7, 16):
                temp_row = list()
                for k in range(0, self.timesteps):
                    temp_row.append(self.train_data_raw[i + j - k])
                
                train_data.append(temp_row)
                train_target.append(self.train_target_raw[i + j])
        
        return train_data, train_target

    def process_test(self):
        
        test_data, test_target = list(), list()

        for i in range(24, len(self.test_data_raw), 24):
            for j in range(7, 16):
                temp_row = list()
                for k in range(0, self.timesteps):
                    temp_row.append(self.test_data_raw[i + j - k])
                
                test_data.append(temp_row)
                test_target.append(self.test_target_raw[i + j])

        return test_data, test_target
    
    
class energyDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target
        
    def __getitem__(self, index):
        return torch.from_numpy(self.data[index]).float(), torch.from_numpy(self.target[index]).float()
        
    def __len__(self):
        return len(self.data)