import pandas as pd
import os

base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, "../data")
output_path = os.path.join(data_path, "nwp")
timestep = 5

if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)

df = pd.read_excel(os.path.join(data_path, "origin/12053002165.xlsx"))

df["power"] = (df["power"] - df["power"].min()) / (df["power"].max() - df["power"].min())
df["rad"] = (df["rad"] - df["rad"].min()) / (df["rad"].max() - df["rad"].min())
df["NWP_Rad"] = (df["NWP_Rad"] - df["NWP_Rad"].min()) / (df["NWP_Rad"].max() - df["NWP_Rad"].min())

train_data_list = df.values[:954]
test_data_list = df.values[954:]

train_data, test_data = dict(), dict()
train_target, test_target = dict(), dict()
train_data["NWP(t)"] = list()
test_data["NWP(t)"] = list()

for i in range(1, timestep):
    train_data[f"NWP(t-{i})"] = list()
    train_data[f"rad(t-{i})"] = list()
    test_data[f"NWP(t-{i})"] = list()
    test_data[f"rad(t-{i})"] = list()

train_target["rad(t)"] = list()
test_target["rad(t)"] = list()

for i in range(0, len(train_data_list), 9):
    for j in range(timestep - 1, 9):
        train_data["NWP(t)"].append(train_data_list[i + j][-2])
        for k in range(1, timestep):
            train_data[f"NWP(t-{k})"].append(train_data_list[i + j - k][-2])
            train_data[f"rad(t-{k})"].append(train_data_list[i + j - k][-1])
        train_target["rad(t)"].append(train_data_list[i + j][-1])

for i in range(0, len(test_data_list), 9):
    for j in range(timestep - 1, 9):
        test_data["NWP(t)"].append(test_data_list[i + j][-2])
        for k in range(1, timestep):
            test_data[f"NWP(t-{k})"].append(test_data_list[i + j - k][-2])
            test_data[f"rad(t-{k})"].append(test_data_list[i + j - k][-1])
        test_target["rad(t)"].append(test_data_list[i + j][-1])

df = pd.DataFrame(train_data)
df.to_csv(os.path.join(output_path, "train_in.csv"), index=False)
df = pd.DataFrame(train_target)
df.to_csv(os.path.join(output_path, "train_out.csv"), index=False)
df = pd.DataFrame(test_data)
df.to_csv(os.path.join(output_path, "test_in.csv"), index=False)
df = pd.DataFrame(test_target)
df.to_csv(os.path.join(output_path, "test_out.csv"), index=False)
