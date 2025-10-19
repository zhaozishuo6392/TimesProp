import numpy as np
def min_trace_reconciliation(y_hat, type='ols', shrinkage_lambda=0.5):
    original_shape = y_hat.shape
    is_sampled = y_hat.ndim == 4
    if is_sampled:
        B, F, N, D = y_hat.shape
        C = D - 1
        y_hat_flat = y_hat.reshape(-1, D)
    else:
        B, F, D = y_hat.shape
        C = D - 1
        y_hat_flat = y_hat.reshape(-1, D)
    S = np.zeros((C+1, C))
    S[0, :] = 1
    for i in range(C):
        S[i + 1, i] = 1
    agg_sum = y_hat_flat[:, 1:].sum(axis=1, keepdims=True)
    residuals = y_hat_flat - np.repeat(agg_sum, D, axis=1)
    if type == 'ols':
        W = np.eye(D)
    elif type == 'var':
        var = np.var(residuals, axis=0)
        W = np.diag(var)
    elif type == 'shr':
        S_cov = np.cov(residuals.T)
        T = np.diag(np.diag(S_cov))
        W = shrinkage_lambda * T + (1 - shrinkage_lambda) * S_cov
    else:
        raise ValueError("type must be 'ols', 'var', or 'shr'")
    W_inv = np.linalg.pinv(W)
    middle = np.linalg.inv(S.T @ W_inv @ S)
    P = S @ middle @ S.T @ W_inv
    y_reconciled_flat = (P @ y_hat_flat.T).T
    if is_sampled:
        y_reconciled = y_reconciled_flat.reshape(B, F, N, D)
    else:
        y_reconciled = y_reconciled_flat.reshape(B, F, D)
    return y_reconciled

def bottom_up_reconciliation(y_hat):
    y_reconciled = y_hat.copy()
    y_reconciled[..., 0] = y_reconciled[..., 1:].sum(axis=-1)
    return y_reconciled

def top_down_reconciliation(y_hat, eps=1e-8):
    y_reconciled = y_hat.copy()
    aggregate = y_hat[..., 0:1]
    children = y_hat[..., 1:]
    total = children.sum(axis=-1, keepdims=True) + eps
    weights = children / total
    reconciled_children = aggregate * weights
    reconciled_aggregate = reconciled_children.sum(axis=-1, keepdims=True)
    y_reconciled[..., 0:1] = reconciled_aggregate
    y_reconciled[..., 1:] = reconciled_children
    return y_reconciled

def top_down_reconciliation2(y_hat, eps=1e-8):
    y_reconciled = y_hat.copy()
    aggregate = y_hat[..., 0:1]
    children = y_hat[..., 1:]
    total = children.sum(axis=-1, keepdims=True) + eps
    weights = children / total
    reconciled_children = aggregate * weights
    reconciled_aggregate = reconciled_children.sum(axis=-1, keepdims=True)
    y_reconciled[..., 0:1] = reconciled_aggregate
    y_reconciled[..., 1:] = reconciled_children
    return y_reconciled

def exact_reconciliation(y_hat):
    original_shape = y_hat.shape
    is_sampled = y_hat.ndim == 4
    if is_sampled:
        B, F, N, D = y_hat.shape
        C = D - 1
        y_children = y_hat.reshape(B * F * N, D)[:, 1:].reshape(B * F * N, C)
    else:
        B, F, D = y_hat.shape
        C = D - 1
        y_children = y_hat.reshape(B * F, D)[:, 1:].reshape(B * F, C)
    S = np.zeros((C + 1, C))
    S[0, :] = 1
    for i in range(C):
        S[i + 1, i] = 1
    y_reconciled_flat = np.einsum('dc,nc->nd', S, y_children)
    if is_sampled:
        y_reconciled = y_reconciled_flat.reshape(B, F, N, D)
    else:
        y_reconciled = y_reconciled_flat.reshape(B, F, D)
    return y_reconciled
