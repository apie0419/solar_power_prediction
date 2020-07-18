import pandas as pd
import xgboost as xgb
import numpy as np
import math, os, pickle
from utils import denorm, rmse, mape
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

base_path = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(base_path, "../data/solar")

train_data = pd.read_excel(os.path.join(data_path, "train_in.xlsx")).values
train_target = pd.read_excel(os.path.join(data_path, "train_out.xlsx"), header=None).values
train_target = train_target[:, 0]
test_data = pd.read_excel(os.path.join(data_path, "test_in.xlsx")).values
test_target = pd.read_excel(os.path.join(data_path, "test_out.xlsx"), header=None).values
test_target = test_target[:, 0]
min_max = pd.read_excel(os.path.join(data_path, "max_min.xls"))
_min, _max = min_max["pmin"][0], min_max["pmax"][0]

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
# preds = denorm(preds, _min, _max)
# target = denorm(np.array(train_target), _min, _max)
target = np.array(train_target)
deno = np.where(target==0, 1, target)
# target = target[:, 0]

print('\nTrain RMSE = ')
print(str(round(math.sqrt(mean_squared_error(target, preds)) * 100., 2)) + "%")
print ("Train MAPE = ")
print("{:.2f}%".format((abs(target - preds)/deno).mean() * 100))

#評估模型

preds = regr.predict(np.array(test_data))
# preds = denorm(preds, _min, _max)
# target = denorm(np.array(test_target), _min, _max)
target = np.array(test_target)
deno = np.where(target==0, 1, target)

print('\nTest RMSE = ')
print(str(round(math.sqrt(mean_squared_error(target, preds)) * 100., 2)) + "%")
print ("Test MAPE = ")
print("{:.2f}%".format((abs(target - preds)/deno).mean() * 100))
# print("{:.2f}%".format((abs(target - preds)/deno).mean() * 100))

preds = denorm(preds, _min, _max)
target = denorm(target, _min, _max)

pd.DataFrame({
    "predict": preds,
    "target": target
}).plot()

plt.savefig(os.path.join(base_path, "Output/power_xgboost_evaluation.png"))
