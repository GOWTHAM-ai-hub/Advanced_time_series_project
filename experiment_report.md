# Experiment Report - Advanced Time Series Forecasting

## Project summary
Implemented Transformer and LSTM-with-Attention models for 1-step-ahead forecasting on a synthetic multivariate hourly dataset (5 features, ~1000 timesteps). Dataset includes trend, daily & weekly seasonality, and noise. Dataset saved at `data/multivariate_synthetic_hourly_5feat_1000steps.csv`.

## Dataset characteristics
- Timesteps: 999
- Features: 5
- Frequency: hourly

## Preprocessing
- Standard scaling (train-mean, train-std)
- Supervised sliding windows (default lookback: 48 steps)

## Models compared
- Transformer encoder (last-step regression)
- LSTM with additive attention over encoder outputs

## Hyperparameters tried
- seq_len: 24, 48, 72
- batch_size: 32, 64
- lr: 1e-3, 5e-4
- d_model (Transformer): 32, 64, 128
- LSTM hidden dim: 32, 64

## Metrics
- MAE, RMSE on validation and test splits

## Suggested next steps
- Extend to multi-step forecasting (direct or recursive)
- Hyperparameter search (Optuna)
- Add exogenous variables or additional synthetic signals
- Compare with classical baselines (ARIMA, Prophet, XGBoost with lag features)

## Results (fill after experiments)
- Transformer: MAE=?, RMSE=?
- LSTM+Attention: MAE=?, RMSE=?

