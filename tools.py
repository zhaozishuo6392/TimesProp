import torch 
import torch.nn.functional as F

def moving_average(x, kernel_size=3):
    """
    x: Tensor of shape (B, H, C)
    kernel_size: odd integer
    """
    padding = kernel_size // 2
    x_padded = F.pad(x.transpose(1, 2), (padding, padding), mode='reflect')  # shape: (B, C, H+2*pad)
    x_smooth = F.avg_pool1d(x_padded, kernel_size=kernel_size, stride=1).transpose(1, 2)  # shape: (B, H, C)
    return x_smooth

def gaussian_kernel1d(kernel_size=5, sigma=1.0):
    half_size = kernel_size // 2
    x = torch.arange(-half_size, half_size + 1, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (x / sigma)**2)
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, -1)  # shape: (1, 1, K)

def gaussian_smoothing(x, kernel_size=5, sigma=1.0):
    """
    x: Tensor of shape (B, H, C)
    """
    B, H, C = x.shape
    kernel = gaussian_kernel1d(kernel_size, sigma).to(x.device)  # (1, 1, K)
    x = x.permute(0, 2, 1)  # (B, C, H)
    x = F.pad(x, (kernel_size // 2, kernel_size // 2), mode='reflect')  # padding for conv1d
    smoothed = F.conv1d(x, kernel.repeat(C, 1, 1), groups=C)  # (B, C, H)
    return smoothed.permute(0, 2, 1)  # (B, H, C)

def exponential_smoothing(x, alpha=0.1):
    """
    x: Tensor of shape (B, H, C)
    """
    x_smooth = torch.zeros_like(x)
    x_smooth[:, 0, :] = x[:, 0, :]
    for t in range(1, x.shape[1]):
        x_smooth[:, t, :] = alpha * x[:, t, :] + (1 - alpha) * x_smooth[:, t - 1, :]
    return x_smooth
