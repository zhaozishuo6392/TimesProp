import numpy as np
import torch.nn.functional as Func
from scipy.stats import dirichlet, norm, beta
import json
import math
import torch
import math
import properscoring as ps

def compute_crps_per_node(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    B, F, N, C = pred.shape
    crps_all = np.zeros((B, F, C))

    for b in range(B):
        for f in range(F):
            for c in range(C):
                # true[b, f, c]: scalar
                # pred[b, f, :, c]: shape (N,)
                crps_all[b, f, c] = ps.crps_ensemble(true[b, f, c], pred[b, f, :, c])

    crps_per_node = crps_all.mean(axis=(0, 1))
    return crps_per_node

def compute_normalized_crps_per_node(pred: np.ndarray, true: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    B, F, N, C = pred.shape
    crps_all = np.zeros((B, F, C))

    for b in range(B):
        for f in range(F):
            for c in range(C):
                crps_all[b, f, c] = crps_ensemble(true[b, f, c], pred[b, f, :, c])

    # Compute std for each node across (B, F)
    std_per_node = np.std(true, axis=(0, 1), keepdims=False)

    # Avoid division by zero
    ncrps_per_node = crps_all.mean(axis=(0, 1)) / (std_per_node + eps)

    return ncrps_per_node



    
def calculate_r_squared(predictions, truths):
    r_squared_values = []
    
    for i in range(predictions.shape[-1]):
        pred = predictions[:, :, i]
        true = truths[:, :, i]
        
        y_mean = np.mean(true)
        
        ss_res = np.sum((true - pred) ** 2)  # Residual sum of squares
        ss_tot = np.sum((true - y_mean) ** 2)  # Total sum of squares
        
        r_squared = 1 - (ss_res / ss_tot)
        r_squared_values.append(r_squared)
    
    return np.array(r_squared_values)


def calculate_smape_per_dim(merged_pred_samples_mean, merged_truth):
    assert merged_pred_samples_mean.shape == merged_truth.shape
    smape_values = compute_smape(merged_pred_samples_mean, merged_truth)
    smape_values_mean = np.mean(smape_values, axis=(0, 1))
    
    return smape_values_mean

def calculate_mae_per_dim(merged_pred_samples_mean, merged_truth):
    assert merged_pred_samples_mean.shape == merged_truth.shape, \
        "Prediction and ground truth shapes must match."

    abs_error = np.abs(merged_pred_samples_mean - merged_truth)
    mae_values_mean = np.mean(abs_error, axis=(0, 1))
    
    return mae_values_mean

def calculate_E_Acc_not_zero(y_true, y_pred):
    y_true_mean = np.sum(y_true, axis=-1)  # Shape: (b, F)

    abs_errors = np.abs(y_pred - y_true)  # Shape: (b, F, C)
    mask = y_true != 0  # Shape: (b, F, C)
    masked_errors = abs_errors * mask

    numerator = np.sum(masked_errors, axis=(1, 2))  # Shape: (b,)
    denominator = 2 * np.sum(y_true_mean, axis=1)  # Shape: (b,)

    metric = 1 - (numerator / denominator)  # Shape: (b,)

    return np.mean(metric)