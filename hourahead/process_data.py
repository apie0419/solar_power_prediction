from scipy.fftpack import fft
import pandas as pd, numpy as np, os

base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, "../data/12053002165")
output_path = os.path.join(base_path, "../data/solar")

if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)

train_data_raw = pd.read_excel(os.path.join(data_path, "train_in.xlsx"), header=None).values
test_data_raw = pd.read_excel(os.path.join(data_path, "test_in.xlsx"), header=None).values
train_target_raw = pd.read_excel(os.path.join(data_path, "train_out.xlsx"), header=None).values
test_target_raw = pd.read_excel(os.path.join(data_path, "test_out.xlsx"), header=None).values

train_data, test_data = list(), list()

timesteps = 24
num_input = 6

"""
Column 0: history power
Column 3: B0062T
"""

for i in range(timesteps, len(train_data_raw)):
    temp_row = list()
    pf = list()
    for j in range(0, timesteps):
        pf.append(train_data_raw[i - j][0])
        
    pf = np.array(pf)
    pf = fft(pf)
    ang_pf = np.angle(pf).tolist()
    abs_pf = np.abs(pf).tolist()
    temp_row.extend(train_data_raw[i].tolist())
    if train_data_raw[i][3] >= train_data_raw[i - 1][3] and train_data_raw[i - 1][3] != 0:
        temp_row[3] = train_data_raw[i][3] - train_data_raw[i - 1][3]
        
    temp_row.extend(abs_pf)
    temp_row.extend(ang_pf)
    
    train_data.append(temp_row)

for i in range(timesteps, len(test_data_raw)):
    
    temp_row = list()
    pf = list()
    for j in range(0, timesteps):
        pf.append(test_data_raw[i - j][0])
    pf = np.array(pf)
    pf = fft(pf)
    ang_pf = np.angle(pf).tolist()
    abs_pf = np.abs(pf).tolist()
    temp_row.extend(test_data_raw[i].tolist())
    temp_row.extend(abs_pf)
    temp_row.extend(ang_pf)
    if test_data_raw[i][3] >= test_data_raw[i - 1][3] and test_data_raw[i - 1][3] != 0:
        temp_row[3] = test_data_raw[i][3] - test_data_raw[i - 1][3]
    test_data.append(temp_row)

train_target = train_target_raw[timesteps:]
test_target = test_target_raw[timesteps:]

train_data_df = pd.DataFrame(train_data)
train_target_df = pd.DataFrame(train_target)
test_data_df = pd.DataFrame(test_data)
test_target_df = pd.DataFrame(test_target)

for i in range(num_input, timesteps * 2 + num_input):
    train_max, train_min = train_data_df[i].max(), train_data_df[i].min()
    test_max, test_min = test_data_df[i].max(), test_data_df[i].min()
    _max, _min = max(train_max, test_max), min(train_min, test_min)
    if _max - _min != 0:
        train_data_df[i] = train_data_df[i] - _min / (_max - _min)
        test_data_df[i] = test_data_df[i] - _min / (_max - _min)


train_data_df.to_excel(os.path.join(output_path, "train_in.xlsx"), index=False, header=None)
test_data_df.to_excel(os.path.join(output_path, "test_in.xlsx"), index=False, header=None)
train_target_df.to_excel(os.path.join(output_path, "train_out.xlsx"), index=False, header=None)
test_target_df.to_excel(os.path.join(output_path, "test_out.xlsx"), index=False, header=None)