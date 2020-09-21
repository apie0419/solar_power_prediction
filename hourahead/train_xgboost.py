from datetime import time
import pandas as pd
import xgboost as xgb
import numpy as np
import os
from utils import Dataset
from matplotlib import pyplot as plt

base_path = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(base_path, "../data/solar")

if not os.path.exists(os.path.join(base_path, "Output")):
    os.mkdir(os.path.join(base_path, "Output"))

timesteps = 24     ### feature dimensions
num_input = 23     ### historic data size
num_output = 1     ### output dimensions

dataset = Dataset(data_path, timesteps)
train_data = np.array(dataset.train_data).reshape(-1, num_input * timesteps)
train_target = np.array(dataset.train_target).reshape(-1, )
test_data = np.array(dataset.test_data).reshape(-1, num_input * timesteps)
test_target = np.array(dataset.test_target).reshape(-1, )

#模型建構

regr = xgb.XGBRegressor(
    gamma=0,
    learning_rate=0.007,
    max_depth=15,
    min_child_weight=10,
    n_estimators=2000,
    reg_alpha=0.5,
    reg_lambda=0.5,
    eval_metric='rmse',
    objective='reg:logistic',
    subsample=0.7,
    n_jobs=-1,
    gpu_id=1,
    tree_method='gpu_hist',
    seed=50
)

# XGBoost training

regr.fit(np.array(train_data), np.array(train_target))

#預測

preds = regr.predict(np.array(train_data))
target = np.array(train_target)

print('\nTrain RMSE = ')
print(str(round(np.sqrt(np.mean(np.square(target - preds))) * 100., 2)) + "%")

#評估模型

preds = regr.predict(np.array(test_data))
target = np.array(test_target)

print('\nTest RMSE = ')
print(str(round(np.sqrt(np.mean(np.square(target - preds))) * 100., 2)) + "%")

pd.DataFrame({
    "predict": preds,
    "target": target
}).plot()

plt.savefig(os.path.join(base_path, "Output/power_xgboost_evaluation.png"))
