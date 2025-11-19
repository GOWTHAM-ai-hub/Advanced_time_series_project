import argparse, torch, os, numpy as np, math
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.data import get_data_loaders
from src.model import TransformerForecaster, LSTMWithAttention

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_path', type=str, default='data/multivariate_synthetic_hourly_5feat_1000steps.csv')
    parser.add_argument('--seq_len', type=int, default=48)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model', type=str, choices=['transformer','lstm'], default='transformer')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    # create model
    # we assume default model hyperparams (match train.py defaults)
    # read n_features from the data
    train_loader, val_loader, test_loader = get_data_loaders(args.data_path, seq_len=args.seq_len, batch_size=args.batch_size)
    n_features = next(iter(train_loader))[0].shape[-1]
    if args.model == 'transformer':
        model = TransformerForecaster(n_features)
    else:
        model = LSTMWithAttention(n_features)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(args.device)
    # evaluate
    ys, preds = [], []
    model.eval()
    with torch.no_grad():
        for x,y in test_loader:
            x = x.to(args.device)
            out = model(x)
            if isinstance(out, tuple): pred = out[0]
            else: pred = out
            ys.append(y.numpy()); preds.append(pred.cpu().numpy())
    ys = np.concatenate(ys); preds = np.concatenate(preds)
    mae = mean_absolute_error(ys, preds); rmse = math.sqrt(mean_squared_error(ys, preds))
    print(f"Test MAE: {mae:.6f}, RMSE: {rmse:.6f}")

if __name__ == '__main__':
    main()
