import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from types import SimpleNamespace
from timesprop import Model
import metrics as metrics
import copy
import os
import argparse
import kit as kit
import torch.nn.functional as F
import Losses as Losses
from torch.optim.lr_scheduler import LambdaLR
from nom_tool import GlobalMinMaxScaler
import ast

warmup_epochs = 0

MODEL_REGISTRATION = {
    "D" : {"LOSS_FUNC" : Losses.dirichlet_loss},
    "GD" : {"LOSS_FUNC" : Losses.gdd_loss},
    "BL" : {"LOSS_FUNC" : Losses.beta_liouville_loss},
    "SD" : {"LOSS_FUNC" : Losses.scaled_dirichlet_loss},
    "SSD" : {"LOSS_FUNC" : Losses.shifted_scaled_dirichlet_loss}
}

def lr_warmup(epoch):
    if epoch < warmup_epochs:
        return float(epoch + 1) / float(warmup_epochs)  # linearly increase
    return 1.0

def gaussian_nll(prob, target):
    mu = prob[..., 0]          # (B, T)
    sigma = prob[..., 1]       # (B, T)
    sigma = sigma.clamp(min=1e-6)
    
    nll = 0.5 * torch.log(2 * torch.pi * sigma**2) + ((target - mu)**2) / (2 * sigma**2)
    return nll.mean()

# === Dataset Class ===
class TimeSeriesDataset(Dataset):
    def __init__(self, data, x_mark, seq_len, pred_len):
        self.data = data
        self.x_mark = x_mark
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len]
        if self.x_mark is not None:
            x_mark = self.x_mark[idx:idx + self.seq_len]
        else:
            x_mark = None
        return x, y, x_mark

def sanitize_and_update_parent(array):
    array = np.copy(array)

    array[array < 0] = 0

    array[:, 0] = array[:, 1:].sum(axis=1)

    return array
    
def load_dataset(df):

    time_col = df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col])
    df[time_col] = df[time_col].dt.tz_localize(None)

    time_features = pd.DataFrame({
        "month": df[time_col].dt.month,
        "day": df[time_col].dt.day,
        "weekday": df[time_col].dt.weekday,
        "hour": df[time_col].dt.hour
    })
    x_mark = torch.tensor(time_features.values, dtype=torch.float32)

    df = df.drop(columns=[time_col])

    data = df.values.astype(np.float32)
    data = sanitize_and_update_parent(data)
    scaler = GlobalMinMaxScaler()
    data = scaler.fit_transform(data)
    data = torch.tensor(data, dtype=torch.float32)

    return data, x_mark, scaler

def crps_sample_loss(y: torch.Tensor, samples: torch.Tensor) -> torch.Tensor:
    B, F, N, D = samples.shape  # D = C+1
    y = y.unsqueeze(2)  # (B, F, 1, C+1)

    term1 = torch.abs(samples - y).mean(dim=2)  # (B, F, C+1)

    samples1 = samples.unsqueeze(3)  # (B, F, N, 1, C+1)
    samples2 = samples.unsqueeze(2)  # (B, F, 1, N, C+1)
    pairwise_diff = torch.abs(samples1 - samples2).mean(dim=(2, 3))  # (B, F, C+1)
    term2 = 0.5 * pairwise_diff
    crps = term1 - term2  # (B, F, C+1)

    if (torch.isnan(crps).any() or torch.isinf(crps).any()):
        print("NaN")

    return crps.mean()
    
def calculate_loss(is_warmup, prob, dist, out, y, lp, dist_type, samples_out):
    dist_loss_func = MODEL_REGISTRATION[dist_type]["LOSS_FUNC"]
    part = y[..., 1:]
    part_clamped = torch.clamp(part, min=0.0)
    sums = part_clamped.sum(dim=-1, keepdim=True)
    eps = 1e-6
    normalized = part_clamped / (sums + eps) # to percent
    
    loss_prob = gaussian_nll(prob, y[..., 0])
    loss_dist = dist_loss_func(normalized, dist)
    loss_crps = crps_sample_loss(y, samples_out)
    
    loss_mse = F.mse_loss(out, y)
    loss_list = []
    if lp[0] == 1:
        loss_list.append(loss_mse*1)
    if lp[1] == 1:
        loss_list.append(loss_prob)
    if lp[2] == 1:
        loss_list.append(loss_dist)

    loss_list.append(loss_crps*1)
    
    loss = sum(loss_list)
    return loss, [loss_mse, loss_prob, loss_dist, loss_crps]

# === Training loop ===
def train_model(model, train_loader, val_loader, optimizer,
                criterion, device, epochs=50, 
                patience=3, warmup_epochs=0, lp= [1, 0, 0],
                dist_type='SSD'):
    best_model = None
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_loss_components = [0.0, 0.0, 0.0, 0.0]
        
        for batch_idx, (x, y, x_mark) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            if x_mark is not None:
                x_mark = x_mark.to(device)

            prob, dist, out, samples_out = model(x, x_mark, None, None)
            loss, loss_components = calculate_loss(epoch < warmup_epochs, prob, dist, out, y, lp, dist_type, samples_out)
            loss = loss
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            total_loss += loss.item()
            for i, comp in enumerate(loss_components):
                train_loss_components[i] += comp.item()
            
        train_loss = total_loss / len(train_loader)
        train_loss_components = [comp / len(train_loader) for comp in train_loss_components]

        # Validation
        model.eval()
        val_loss = 0
        val_loss_components = [0.0, 0.0, 0.0, 0.0]
        
        with torch.no_grad():
            for batch_idx, (x, y, x_mark) in enumerate(val_loader):
                if x_mark is not None:
                    x_mark = x_mark.to(device)
                x, y = x.to(device), y.to(device)
                prob, dist, out, samples_out = model(x, x_mark, None, None)
                loss, loss_components = calculate_loss(epoch < warmup_epochs, prob, dist, out, y, lp, dist_type, samples_out)
                val_loss += loss.item()
                
                for i, comp in enumerate(loss_components):
                    val_loss_components[i] += comp.item()

                
        val_loss /= len(val_loader)
        val_loss_components = [comp / len(val_loader) for comp in val_loss_components]

        is_warmup = epoch < warmup_epochs
        warmup_status = " (Warmup)" if is_warmup else ""

        train_components_str = ", ".join([f"{comp:.4f}" for comp in train_loss_components])
        val_components_str = ", ".join([f"{comp:.4f}" for comp in val_loss_components])
        print(f"Train: {train_loss:.4f}, Components: [{train_components_str}]", end='\t')
        print(f"Val: {val_loss:.4f}, Components: [{val_components_str}]")

        if not is_warmup:
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

    best_model_path = '/workspace/saved_model.pth'
    
    torch.save(best_model, best_model_path)
    model.load_state_dict(torch.load(best_model_path))
    return model

# === Evaluation ===
def evaluate_model(model, test_loader, scaler, device):
    model.eval()
    preds = []
    samples = []
    
    print('\nEvaluating model...')
    with torch.no_grad():
        for batch_idx, (x, _, x_mark) in enumerate(test_loader):
            x = x.to(device)
            x_mark = x_mark.to(device)
            prob, dist, out, samples_out = model(x, x_mark, None, None)
            
            out = out.cpu().numpy() # (b, F, C+1)
            out = scaler.inverse_transform(out.reshape(-1, out.shape[-1])).reshape(out.shape)
            preds.append(out)

            samples_out = samples_out.cpu().numpy() # (b, F, 100, C+1)
            samples_out = scaler.inverse_transform(samples_out.reshape(-1, samples_out.shape[-1])).reshape(samples_out.shape)
            samples.append(samples_out)
            
    return np.concatenate(preds, axis=0), np.concatenate(samples, axis=0)

# === Main ===
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model with batch size and file path")
    parser.add_argument("--path", type=str, required=True, help="File path relative to /workspace/data")
    parser.add_argument("--H", type=int, default=168, help="History")
    parser.add_argument("--repeat", type=int, default=5, help="Number of repeated training runs")
    parser.add_argument("--load", action="store_true", help="Whether to load model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lp", type=int, nargs='+', default=[1, 1, 1], help="Loss")
    parser.add_argument("--p", type=int, default=5, help="Patience")
    parser.add_argument("--dist", type=str, required=True, help="Dist type")
    parser.add_argument("--MHA", type=int, default=0, help="Period Type")
    parser.add_argument("--d_model", type=int, default=32, help="d_model")

    parser.add_argument("--period", type=ast.literal_eval, help="Period List")
    parser.add_argument("--part", type=ast.literal_eval, help="Part List")
    

    args = parser.parse_args()
    full_path = os.path.join("/workspace/data", args.path)
    
    method_names = {
        # "Unreconciled": None,
        # "MinT": lambda x: kit.min_trace_reconciliation(x),
        # "Bottom-Up": kit.bottom_up_reconciliation,
        "Top-Down": kit.top_down_reconciliation,
        # "MinT-ols": lambda x: kit.min_trace_reconciliation(x, 'ols'),
        # "MinT-var": lambda x: kit.min_trace_reconciliation(x, 'var'),
        # "MinT-shr": lambda x: kit.min_trace_reconciliation(x, 'shr'),
        # "ERM": lambda x: kit.exact_reconciliation(x)
    }
    results = {
        name: {"smape": [], "r2": [], "eacc": [], "crps": [], "mae_children": []}
        for name in method_names
    }

    print(f'Starting {args.repeat} training repeats...')
    
    for repeat in range(args.repeat):
        print(f"======== Training Repeat {repeat + 1}/{args.repeat} ========")
        
        df = pd.read_csv(full_path)
        child_num = df.shape[1]-1

        configs = SimpleNamespace(
            seq_len=args.H,
            pred_len=24,
            d_model=args.d_model, # 64
            enc_in=child_num,
            c_out=child_num,
            embed='timeF',
            freq='h',
            dropout=0.1,
            dist_type=args.dist,
            period_list = args.period,
            part_list = args.part,
            mha = args.MHA
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        df = pd.read_csv(full_path)
        data, x_mark, scaler = load_dataset(df)

        total_len = data.shape[0]
        train_len = int(total_len * 0.7)
        val_len = int(total_len * 0.1)
        
        print(f"total_len: {total_len}")
        print(f"train_len: {train_len}")
        print(f"val_len: {val_len}")

        train_data = data[:train_len]
        val_data = data[train_len:train_len + val_len]
        test_data = data[train_len + val_len:]

        train_mark = x_mark[:train_len] if x_mark is not None else None
        val_mark = x_mark[train_len:train_len + val_len] if x_mark is not None else None
        test_mark = x_mark[train_len + val_len:] if x_mark is not None else None

        train_dataset = TimeSeriesDataset(train_data, train_mark, configs.seq_len, configs.pred_len)
        val_dataset = TimeSeriesDataset(val_data, val_mark, configs.seq_len, configs.pred_len)
        test_dataset = TimeSeriesDataset(test_data, test_mark, configs.seq_len, configs.pred_len)

        batch_size = args.batch_size
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        model = Model(configs).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = LambdaLR(optimizer, lr_warmup)

        criterion = nn.MSELoss()

        print(f"Using batch size: {batch_size}")

        if args.load:
            best_model_path = '/workspace/timesnet-prob2/saved_model.pth'
            model.load_state_dict(torch.load(best_model_path))
        else:
            model = train_model(model, train_loader, val_loader, optimizer, criterion, device, 
                              epochs=50, patience=args.p, warmup_epochs=warmup_epochs, lp=args.lp, dist_type=args.dist)
        

        preds, samples = evaluate_model(model, test_loader, scaler, device)
        preds = samples.mean(axis=2)

        y_true = []
        with torch.no_grad():
            for _, y, _ in test_loader:
                y = y.cpu().numpy()
                y = scaler.inverse_transform(y.reshape(-1, y.shape[-1])).reshape(y.shape)
                y_true.append(y)
        y_true = np.concatenate(y_true, axis=0)

        show_samples = kit.top_down_reconciliation2(samples)
        out_samples = samples

        batch_ = 0

        preds_out = show_samples.mean(axis=2)
        preds_out = kit.top_down_reconciliation(preds)
        for app in range(y_true.shape[2]):
            for t in range(y_true.shape[1]):
                print(f"{y_true[batch_, t, app]:.3f}", end='\t')
            print('')
            for t in range(y_true.shape[1]):
                print(f"{preds_out[batch_, t, app]:.3f}", end='\t')
            print('')
            
        
        for method, func in method_names.items():
            pred = preds if func is None else func(preds)
            outs = out_samples if func is None else func(out_samples)

            mae = metrics.calculate_mae_per_dim(pred, y_true)
            smape = metrics.calculate_smape_per_dim(pred, y_true)
            mae_children = mae[1:]
            smape = smape[0]
            r2 = metrics.calculate_r_squared(pred, y_true)[0]
            eacc = metrics.calculate_E_Acc_not_zero(y_true[..., 1:], pred[..., 1:])
            crps_all_nodes = metrics.compute_crps_per_node(outs, y_true)
    
            results[method]["smape"].append(smape)
            results[method]["r2"].append(r2)
            results[method]["eacc"].append(eacc)
            results[method]["crps"].append(crps_all_nodes)
            results[method]["mae_children"].append(mae_children)

        del model, optimizer, train_loader, val_loader, test_loader
        del train_dataset, val_dataset, test_dataset
        del preds, samples, y_true
        print(f"Completed repeat {repeat + 1}/{args.repeat}")


    print("======== FINAL RESULTS ========")
    print(args.path)
    print("\033[34m", end='')
    for method, res in results.items():
        smape_arr = np.array(res["smape"])
        r2_arr = np.array(res["r2"])
        eacc_arr = np.array(res["eacc"])
    
        suffix = f"Sep77-{method}"
        print(f"{suffix}\t{smape_arr.mean():.3f}±{smape_arr.std():.3f}\t{r2_arr.mean():.3f}±{r2_arr.std():.3f}\t{eacc_arr.mean():.3f}±{eacc_arr.std():.3f}")
    # print('')
    for method, res in results.items():
        crps_arr = np.array(res["crps"])
        for i in range(0, crps_arr.shape[1]):
            print(f"{crps_arr[:, i].mean():.3f}±{crps_arr[:, i].std():.3f}", end='\t')

        for crps in crps_arr:
            print(f"{crps.mean():.3f}±{crps.std():.3f}", end='\t')
        print('')
    print('')
    print("\033[0m", end='')
