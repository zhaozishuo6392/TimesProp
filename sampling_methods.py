import torch
import torch.nn as nn
import torch.nn.functional as functional

def generate_normal_samples(params: torch.Tensor, n_samples: int = 100) -> torch.Tensor:
    num_samples = n_samples
    if params.shape[-1] != 2:
        raise ValueError("Last dimension must be 2 (mean and std).")
    B, F, _ = params.shape
    mean = params[:, :, 0]
    std = params[:, :, 1]
    std = torch.clamp(std, min=1e-6)
    normal = torch.distributions.Normal(mean, std)
    samples = normal.rsample((num_samples,))
    samples = samples.permute(1, 2, 0).unsqueeze(-1)
    return samples

def sample_dirichlet(alpha: torch.Tensor, n_samples: int = 100):
    num_samples = n_samples
    eps = 1e-6
    alpha_expanded = alpha.unsqueeze(-1).expand(-1, -1, -1, num_samples)
    gamma_samples = torch.distributions.Gamma(alpha_expanded + eps, torch.ones_like(alpha_expanded)).rsample()
    samples = gamma_samples / (gamma_samples.sum(dim=2, keepdim=True) + eps)
    samples = samples.permute(0, 1, 3, 2)
    return samples

def sample_gd(params: torch.Tensor, n_samples: int = 100):
    num_samples = n_samples
    B, F, two_c_minus_2 = params.shape
    C_minus_1 = two_c_minus_2 // 2
    C = C_minus_1 + 1
    alpha_beta = params.view(B, F, C_minus_1, 2)
    alpha = alpha_beta[..., 0]
    beta = alpha_beta[..., 1]
    alpha_exp = alpha.unsqueeze(2).expand(B, F, num_samples, C_minus_1)
    beta_exp = beta.unsqueeze(2).expand(B, F, num_samples, C_minus_1)
    gamma1 = torch.distributions.Gamma(alpha_exp, torch.ones_like(alpha_exp)).rsample()
    gamma2 = torch.distributions.Gamma(beta_exp, torch.ones_like(beta_exp)).rsample()
    Z = gamma1 / (gamma1 + gamma2)
    remaining = torch.cumprod(1 - Z + 1e-7, dim=-1)
    remaining = torch.cat([torch.ones_like(remaining[..., :1]), remaining[..., :-1]], dim=-1)
    X = Z * remaining
    sum_X = X.sum(dim=-1, keepdim=True)
    X_last = 1.0 - sum_X
    samples = torch.cat([X, X_last], dim=-1)
    samples = samples
    return samples

def sample_bl(params: torch.Tensor, n_samples: int = 100):
    B, F, total_params = params.shape
    K = total_params - 1
    alpha_dirichlet = params[:, :, :K]
    alpha_beta = params[:, :, -2]
    beta_beta = params[:, :, -1]
    gamma_alpha = alpha_dirichlet[..., :-1].unsqueeze(2).expand(B, F, n_samples, K-1)
    gamma_sample = torch.distributions.Gamma(gamma_alpha, torch.ones_like(gamma_alpha)).rsample()
    dirichlet_part = gamma_sample / gamma_sample.sum(dim=-1, keepdim=True)
    alpha_g = torch.distributions.Gamma(alpha_beta, torch.ones_like(alpha_beta)).rsample((n_samples,))
    beta_g = torch.distributions.Gamma(beta_beta, torch.ones_like(beta_beta)).rsample((n_samples,))
    u = alpha_g / (alpha_g + beta_g)
    u = u.permute(1, 2, 0)
    scaled = u.unsqueeze(-1) * dirichlet_part
    phi_k = 1.0 - scaled.sum(dim=-1, keepdim=True)
    samples = torch.cat([scaled, phi_k], dim=-1)
    return samples

def scaled_dirichlet_sampling(params, n_samples=100):
    num_samples = n_samples
    b, F, D2 = params.shape
    D = D2 // 2
    alpha = params[:, :, :D].clone()
    beta = params[:, :, D:].clone()
    gamma_samples = torch.distributions.Gamma(alpha, beta).rsample((num_samples,))
    samples = gamma_samples / gamma_samples.sum(dim=-1, keepdim=True)
    samples = samples.permute(1, 2, 0, 3)
    return samples

def sample_ssd_batch(params_batch, n_samples=100):
    num_samples = n_samples
    b, F, param_dim = params_batch.shape
    C = (param_dim - 1) // 2
    if (param_dim - 1) % 2 != 0:
        raise ValueError("Invalid parameter shape. Expected 2*C+1 dimensions.")
    alpha = params_batch[..., :C]
    scale_factors = params_batch[..., C:-1]
    a = params_batch[..., -1:]
    device = params_batch.device
    alpha_expanded = alpha.unsqueeze(2).expand(b, F, num_samples, C)
    scale_factors_expanded = scale_factors.unsqueeze(2).expand(b, F, num_samples, C)
    a_expanded = a.unsqueeze(2).expand(b, F, num_samples, 1)
    gamma_samples = torch.distributions.Gamma(concentration=alpha_expanded, rate=torch.ones_like(alpha_expanded)).rsample()
    dir_samples = gamma_samples / gamma_samples.sum(dim=-1, keepdim=True)
    powered_samples = torch.pow(dir_samples, a_expanded)
    perturbed_samples = powered_samples * scale_factors_expanded
    normalized_samples = perturbed_samples / perturbed_samples.sum(dim=-1, keepdim=True)
    return normalized_samples
