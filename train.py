import argparse, os, torch, math
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from src.data import get_data_loaders
from src.model import TransformerForecaster, LSTMWithAttention
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    running_loss = 0.0
    for x,y in loader:
        x = x.to(device); y = y.to(device)
        opt.zero_grad()
        if hasattr(model, '__call__'):
            out = model(x)
            if isinstance(out, tuple):
                pred = out[0]
            else:
                pred = out
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        running_loss += loss.item() * x.size(0)
    return running_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device); y = y.to(device)
            out = model(x)
            if isinstance(out, tuple):
                pred = out[0]
            else:
                pred = out
            ys.append(y.cpu().numpy()); preds.append(pred.cpu().numpy())
    ys = np.concatenate(ys); preds = np.concatenate(preds)
    mae = mean_absolute_error(ys, preds); rmse = math.sqrt(mean_squared_error(ys, preds))
    return mae, rmse, ys, preds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/multivariate_synthetic_hourly_5feat_1000steps.csv')
    parser.add_argument('--model', type=str, choices=['transformer','lstm'], default='transformer')
    parser.add_argument('--seq_len', type=int, default=48)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    train_loader, val_loader, test_loader = get_data_loaders(args.data_path, seq_len=args.seq_len, batch_size=args.batch_size)
    n_features = next(iter(train_loader))[0].shape[-1]

    device = torch.device(args.device)
    if args.model == 'transformer':
        model = TransformerForecaster(n_features).to(device)
    else:
        model = LSTMWithAttention(n_features).to(device)

    opt = Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val = float('inf')
    for epoch in range(1, args.epochs+1):
        train_loss = train_one_epoch(model, train_loader, opt, loss_fn, device)
        val_mae, val_rmse, _, _ = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.6f} | val_mae={val_mae:.6f} | val_rmse={val_rmse:.6f}")
        # save best by val_rmse
        if val_rmse < best_val:
            best_val = val_rmse
            torch.save({'model_state_dict': model.state_dict(), 'args': vars(args)}, os.path.join(args.checkpoint_dir, 'best_model.pt'))

    # final test eval
    test_mae, test_rmse, ys, preds = evaluate(model, test_loader, device)
    print(f"Test results - MAE: {test_mae:.6f}, RMSE: {test_rmse:.6f}")
    # save last checkpoint
    torch.save({'model_state_dict': model.state_dict(), 'args': vars(args)}, os.path.join(args.checkpoint_dir, 'last_model.pt'))

if __name__ == '__main__':
    main()
