import pandas as pd
import xgboost as xgb
import numpy as np
import math, os, pickle
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

base_path = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(base_path, "../data/solar")

train_data = pd.read_csv(os.path.join(data_path, "train_in.csv")).values
train_target = pd.read_csv(os.path.join(data_path, "train_out.csv")).values
train_target = train_target[:, 0]
test_data = pd.read_csv(os.path.join(data_path, "test_in.csv")).values
test_target = pd.read_csv(os.path.join(data_path, "test_out.csv")).values
test_target = test_target[:, 0]

#模型建構

regr = xgb.XGBRegressor(
    gamma=0,
    learning_rate=0.01,
    max_depth=20,
    min_child_weight=10,
    n_estimators=2000,
    reg_alpha=0.5,
    reg_lambda=0.7,
    eval_metric='rmse',
    objective='reg:logistic',
    subsample=0.8,
    seed=50
)

# XGBoost training

regr.fit(np.array(train_data), np.array(train_target))

#預測

preds = regr.predict(np.array(train_data))
target = np.array(train_target)
deno = np.where(target==0, 1, target)
# target = target[:, 0]

print('\nTrain RMSE = ')
print(math.sqrt(mean_squared_error(target, preds)))
print ("Train MAPE = ")
print("{:.2f}%".format((abs(target - preds)/deno).mean() * 100))

#評估模型

preds = regr.predict(np.array(test_data))
target = np.array(test_target)
deno = np.where(target==0, 1, target)

print('\nTest RMSE = ')
print(math.sqrt(mean_squared_error(target, preds)))
print ("Test MAPE = ")
print("{:.2f}%".format((abs(target - preds)/deno).mean() * 100))

pd.DataFrame({
    "predict": preds,
    "target": target
}).plot()

plt.savefig(os.path.join(base_path, "Output/power_xgboost_evaluation.png"))
