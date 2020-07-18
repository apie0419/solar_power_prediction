import pandas as pd
import xgboost as xgb
import os, pickle

base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, "../data")
output_path = os.path.join(data_path, "solar")
timestep = 5

if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)

regr = pickle.load(open(os.path.join(base_path, "nwp.pickle.dat"), "rb"))

train_df = pd.read_csv(os.path.join(data_path, "nwp/train_in.csv"))
test_df = pd.read_csv(os.path.join(data_path, "nwp/test_in.csv"))
train_data = train_df.values
test_data = test_df.values

df = pd.read_excel(os.path.join(data_path, "origin/12053002165.xlsx"))
_min, _max  = df["power"].min(), df["power"].max()
df["power"] = (df["power"] - _min) / (_max - _min)

train_data_list = df.values[:954]
test_data_list = df.values[954:]

train_target, test_target = pd.DataFrame(columns=["power(t)"]), pd.DataFrame(columns=["power(t)"])
power = list()

for i in range(0, len(train_data_list), 9):
    for j in range(timestep - 1, 9):
        train_target = train_target.append({"power(t)": train_data_list[i + j][-3]}, ignore_index=True)
        power.append(train_data_list[i + j - 1][-3])

train_df["power(t-1)"] = power

power = list()

for i in range(0, len(test_data_list), 9):
    for j in range(timestep - 1, 9):
        test_target = test_target.append({"power(t)": test_data_list[i + j][-3]}, ignore_index=True)
        power.append(test_data_list[i + j - 1][-3])

test_df["power(t-1)"] = power

train_preds = regr.predict(train_data)
train_df["NWP(t)"] = train_preds

test_preds = regr.predict(test_data)
test_df["NWP(t)"] = test_preds

train_df.to_csv(os.path.join(output_path, "train_in.csv"), index=False)
test_df.to_csv(os.path.join(output_path, "test_in.csv"), index=False)
train_target.to_csv(os.path.join(output_path, "train_out.csv"), index=False)
test_target.to_csv(os.path.join(output_path, "test_out.csv"), index=False)
pd.DataFrame({"min": [_min], "max": [_max]}).to_csv(os.path.join(output_path, "min_max.csv"), index=False)
