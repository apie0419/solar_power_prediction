# Solar Power Forcasting

## Environment Installation

- Ubuntu 18.04

```
conda env create -f environment.yml -n <env_name>
```

## Run Code Step by Step

1. Produce Training Data, Apply CWT and PCA.

```
python process_data.py
```

2. Train Model to Predict Power

- TPA-LSTM

```
python train_tpalstm.py
```

- TCN

```
python train_tcn.py
```

- XGBoost

```
python train_xgboost.py
```