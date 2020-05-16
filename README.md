# Solar Power Forcasting

## Environment Installation

- Ubuntu 18.04

```
conda env create -f environment.yml -n <env_name>
```

## Run Code Step by Step

1. Process Rad Prediction Training Data

```
python process_rad_data.py
```

2. Train Model to Predict Rad

```
python train_rad_xgboost.py
```

3. Process Solar Power Prediction Training Data

```
python process_power_data.py
```

4. Train Model to Predict Solar Power

```
python train_power_xgboost.py
```

