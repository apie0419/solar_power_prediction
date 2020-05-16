import os
import pandas as pd
import numpy as np

class Dataset(object):
    
    def __init__(self, path, timesteps):
        self.data_path = path
        self.timesteps = timesteps
        self.train_data_raw = pd.read_excel(os.path.join(path, "hour_ahead/train_in.xlsx")).values
        self.train_target_raw = pd.read_excel(os.path.join(path, "hour_ahead/train_out.xlsx")).values
        self.test_data_raw = pd.read_excel(os.path.join(path, "hour_ahead/test_in.xlsx")).values
        self.test_target_raw = pd.read_excel(os.path.join(path, "hour_ahead/test_out.xlsx")).values
        self.train_data, self.train_target = self.process_train()
        self.test_data, self.test_target = self.process_test()
        min_max = pd.read_excel(os.path.join(self.data_path, "hour_ahead/max_min.xls"))
        self._min = float(min_max["pmin"][0])
        self._max = float(round(min_max["pmax"][0], 2))
    
    def process_train(self):
        train_data, temp_row = list(), list()
        for i in range(len(self.train_data_raw) - (self.timesteps - 1)):
            for j in range(self.timesteps):
                temp_row.append(list(self.train_data_raw[i + j]))
            
            train_data.append(temp_row)
            temp_row = list()
        
        train_target = self.train_target_raw[self.timesteps-1:]

        return train_data, train_target

    def process_test(self):
        test_data, temp_row = list(), list()
        for i in range(len(self.test_data_raw) - (self.timesteps - 1)):
            for j in range(self.timesteps):
                temp_row.append(list(self.test_data_raw[i + j]))
            
            test_data.append(temp_row)
            temp_row = list()
        
        test_target = self.test_target_raw[self.timesteps-1:]

        return test_data, test_target

    def append(self, dataset):
        pass
    
    
