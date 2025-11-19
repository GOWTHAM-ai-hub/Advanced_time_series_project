# Advanced Time Series Forecasting - Project Bundle

This repository contains a complete starter project for the "Advanced Time Series Forecasting with Deep Learning and Attention Mechanisms" assignment.
It includes:
- A synthetic multivariate dataset (5 features, 1000 hourly timesteps)
- Data pipeline and PyTorch dataloaders
- Two model options: a Transformer-based encoder and an LSTM with attention
- Training and evaluation scripts, plus model saving and example inference code
- A short report template to document experiments and results

**Files of interest:**
- `data/multivariate_synthetic_hourly_5feat_1000steps.csv` - dataset
- `src/data.py` - data-generation/loader utilities
- `src/model.py` - model definitions (Transformer & LSTM+Attention)
- `src/train.py` - training loop with hyperparameter options & checkpoint saving
- `src/evaluate.py` - evaluation metrics (MAE, RMSE) on test set
- `report/experiment_report.md` - report template

**How to run (example):**
1. Create a virtual environment and install requirements: `pip install -r requirements.txt`
2. Train (Transformer model): `python src/train.py --model transformer --data_path data/multivariate_synthetic_hourly_5feat_1000steps.csv`
3. Evaluate: `python src/evaluate.py --checkpoint checkpoints/best_model.pt --data_path data/...`

See `src/train.py --help` for hyperparameters (sequence length, batch size, learning rate, epochs, etc.).
