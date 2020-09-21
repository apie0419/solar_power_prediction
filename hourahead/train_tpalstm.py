from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from utils import Dataset, energyDataset
from tpalstm import TPALSTM

import torch, random
import os, pandas as pd, numpy as np, torch.nn.functional as F

GPU                = 0
batch_size         = 64
hidden_units       = 32
base_learning_rate = 0.005
n_layers           = 5
epoches            = 200
output_dims        = 1           ### output dimensions
timesteps          = 24          ### historic data size
num_input          = 23          ### feature dimensions
seed               = 0

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.set_device(GPU)
torch.backends.cudnn.deterministic=True

base_path = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(base_path, "../data/solar")

dataset = Dataset(data_path, timesteps)

train_data, train_target = np.array(dataset.train_data).astype(np.float32), np.array(dataset.train_target).astype(np.float32)

kf = KFold(n_splits=5, shuffle=True)

best_model = None
best_val_loss = 100
best_losses = None
fold = 1

for train_index, val_index in kf.split(train_data):
    
    t_data = train_data[train_index]
    t_target = train_target[train_index]
    v_data = train_data[val_index]
    v_target = train_target[val_index]

    trainloader = DataLoader(energyDataset(t_data, t_target), batch_size=batch_size, shuffle=True, num_workers=0)

    model = TPALSTM(num_input, output_dims, hidden_units, timesteps, n_layers)
    model = model.cuda()

    learning_rate = base_learning_rate
    optimizer = Adam(model.parameters(), lr=learning_rate)

    pbar = tqdm(range(1, epoches+1))

    model.train()

    losses = list()

    for epoch in pbar:
        if epoch % 100 == 0:
            learning_rate *= 0.7
            optimizer = Adam(model.parameters(), lr=learning_rate)

        train_loss = 0.
        for data, target in trainloader:
            data, target = data.cuda(), target.cuda()
            preds = model(data)
            loss = F.mse_loss(preds, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
        train_loss /= len(trainloader.dataset)
        train_loss = round(np.sqrt(train_loss), 4)
        losses.append(train_loss)
        pbar.set_description(f"Fold {fold}, Epoch {epoch}, Loss: {train_loss}")
    
    model.eval()

    with torch.no_grad():
        val_data = torch.from_numpy(np.array(v_data).astype(np.float32)).float()
        val_target = np.array(v_target).astype(np.float32)
        val_data = val_data.cuda()
        val_preds = model(val_data)
        val_preds = val_preds.data.cpu().numpy()
        val_loss = round(np.sqrt(np.mean(np.square(val_target - val_preds))) * 100., 2)
    
    print (f"Validation Loss: {val_loss}%")

    if val_loss < best_val_loss:
        best_model = model
        best_val_loss = val_loss
        best_losses = losses
    
    fold += 1

best_model.eval()

with torch.no_grad():
    train_data = torch.from_numpy(np.array(dataset.train_data).astype(np.float32)).float()
    train_target = np.array(dataset.train_target).astype(np.float32)
    train_data = train_data.cuda()
    train_preds = best_model(train_data)
    train_preds = train_preds.data.cpu().numpy()
    train_loss = round(np.sqrt(np.mean(np.square(train_target - train_preds))) * 100., 2)
    test_data = torch.from_numpy(np.array(dataset.test_data).astype(np.float32)).float()
    test_target = np.array(dataset.test_target).astype(np.float32)
    test_data = test_data.cuda()
    test_preds = best_model(test_data)
    test_preds = test_preds.data.cpu().numpy()
    test_loss = round(np.sqrt(np.mean(np.square(test_target - test_preds))) * 100., 2)

print (f"Train RMSE Loss: {train_loss}%, Test RMSE Loss: {test_loss}%")

if not os.path.exists(os.path.join(base_path, "Output")):
    os.mkdir(os.path.join(base_path, "Output"))

pd.DataFrame({
    "loss": best_losses
}).plot()

plt.savefig(os.path.join(base_path, "Output/tpalstm_loss.png"))

print("Optimization Finished!")

pd.DataFrame({
    "predict": test_preds[:, 0],
    "target": test_target[:, 0]
}).plot()

plt.savefig(os.path.join(base_path, "Output/tpalstm_evaluation.png"))
