import torch
import torch.nn as nn
import torch.nn.functional as F

class PeriodTransformerEncoder(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dim_feedforward=model_dim * 4, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(model_dim, input_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, model_dim))

    def forward(self, segments):
        B, N, P_num, Period = segments.shape
        x = segments.permute(0, 2, 1, 3).reshape(B * P_num, N, Period).transpose(1, 2)
        x = self.input_proj(x)
        pos = self.pos_embedding[:, :Period, :]
        x = x + pos
        x = self.transformer(x)
        x = self.output_proj(x)
        x = x.transpose(1, 2).reshape(B, P_num, N, Period).permute(0, 2, 1, 3).contiguous()
        return x

def split_tensor_last_dim(tensor: torch.Tensor, n: int):
    A, B, C, D = tensor.shape
    base = D // n
    remainder = D % n
    split_sizes = [base] * n
    for i in range(remainder):
        split_sizes[i] += 1
    chunks = torch.split(tensor, split_sizes, dim=-1)
    return list(chunks)

def compute_period_influence(x: torch.Tensor, period_list: list[int]) -> torch.Tensor:
    B, T, C = x.shape
    x_c0 = x[..., 0]
    influences = []
    for p in period_list:
        usable_len = (T // p) * p
        x_cut = x_c0[:, :usable_len]
        x_reshaped = x_cut.reshape(B, -1, p)
        mean_period = x_reshaped.mean(dim=1, keepdim=True)
        mse = ((x_reshaped - mean_period) ** 2).mean(dim=[1, 2])
        influences.append(-mse)
    influence_tensor = torch.stack(influences, dim=1)
    influence_weights = F.softmax(influence_tensor, dim=1)
    return influence_weights

class PeriodAwareConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        kernel_list = [(1, 3), (3, 1), (3, 3), (1, 5), (5, 1)]
        self.convs = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=(k[0] // 2, k[1] // 2)) for k in kernel_list])
        self.num_branches = len(self.convs)
        self.attn = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(out_channels, self.num_branches, kernel_size=1), nn.Softmax(dim=1))
        self.act = nn.GELU()
        self.norm = nn.GroupNorm(4, out_channels)

    def forward(self, x):
        outs = [conv(x) for conv in self.convs]
        outs = torch.stack(outs, dim=1)
        weights = self.attn(torch.mean(outs, dim=1))
        weights = weights.unsqueeze(2)
        out = torch.sum(outs * weights, dim=1)
        out = self.norm(self.act(out))
        return out

def split_segments_from_shift_with_padding_fast(signal: torch.Tensor, period: int, shift: int, fill_value: float = -1.0):
    T, C = signal.shape
    device = signal.device
    dtype = signal.dtype

    s = shift % period

    need_left = (period - s) % period
    if need_left == 0:
        start = max(0, shift - period)
        first_segment = signal[start:shift]
        if first_segment.shape[0] < period:
            pad_left_len = period - first_segment.shape[0]
            pad_left = torch.full((pad_left_len, C), fill_value, dtype=dtype, device=device)
            first_segment = torch.cat([pad_left, first_segment], dim=0)
        else:
            pad_left_len = 0
    else:
        pad_left_len = max(0, need_left - max(0, shift))
        take_len = period - pad_left_len
        
        first_right = signal[max(0, shift - take_len):shift]
        if first_right.shape[0] < take_len:
            
            pad_left_len = period - first_right.shape[0]
        pad_left = torch.full((pad_left_len, C), fill_value, dtype=dtype, device=device)
        first_segment = torch.cat([pad_left, first_right], dim=0)

    if first_segment.shape[0] != period:
        if first_segment.shape[0] > period:
            first_segment = first_segment[-period:]
        else:
            need = period - first_segment.shape[0]
            first_segment = torch.cat([torch.full((need, C), fill_value, dtype=dtype, device=device), first_segment], dim=0)

    rest = signal[shift:]
    total_rest = rest.shape[0]
    num_segments = (total_rest + period - 1) // period if total_rest > 0 else 0
    pad_right_len = num_segments * period - total_rest if num_segments > 0 else 0

    if pad_right_len > 0:
        pad_right = torch.full((pad_right_len, C), fill_value, dtype=dtype, device=device)
        rest = torch.cat([rest, pad_right], dim=0)

    segments_rest = rest.view(num_segments, period, C) if num_segments > 0 else rest.new_zeros((0, period, C))

    stacked = torch.cat([first_segment.unsqueeze(0), segments_rest], dim=0)

    mask = stacked != fill_value
    complete_mask = mask.all(dim=2).all(dim=1)  # (N,)
    if complete_mask.any():
        mean_profile = stacked[complete_mask].mean(dim=0, keepdim=True)
    else:
        mean_profile = torch.zeros((1, period, C), dtype=dtype, device=device)
    stacked = torch.where(mask, stacked, mean_profile.expand_as(stacked))

    pad_left_len = int(pad_left_len)
    pad_right_len = int(pad_right_len)

    return stacked, pad_left_len, pad_right_len

def compute_shift_to_center_max2(signal: torch.Tensor, period: int, top_n: int = 3) -> int:
    device = signal.device
    T, C = signal.shape
    assert T >= period, "Signal too short for given period"

    signal_c0 = signal[:, 0]  # (T,)
    num_segments = T // period
    segments = signal_c0[:num_segments * period].reshape(num_segments, period)

    peak_vals, peak_idxs = torch.max(segments, dim=1)  # (num_segments,)
    top_vals, top_indices = torch.topk(peak_vals, k=min(top_n, num_segments))
    selected_peak_positions = torch.gather(peak_idxs, 0, top_indices)  # (top_n,)

    angles = selected_peak_positions.float() * (2 * torch.pi / period)  # (top_n,)
    x = torch.cos(angles).mean()
    y = torch.sin(angles).mean()
    avg_angle = torch.atan2(y, x)

    avg_peak_pos = (avg_angle * period / (2 * torch.pi)) % period
    avg_peak_pos = torch.round(avg_peak_pos).long()

    center = period // 2
    raw_shift = avg_peak_pos - center
    shift = raw_shift if raw_shift >= 0 else raw_shift + period

    return shift.item()

def process_signal_to_2d(signal: torch.Tensor, time_list: torch.Tensor, period: int, pred_len: int = 24, is_full: bool = True, num=8):
    if num == 8:
        shift = compute_shift_to_center_max2(signal, period=period, top_n=1)
    if not is_full:
        pad = torch.full((pred_len, signal.shape[1]), -1.0, dtype=signal.dtype, device=signal.device)
        signal = torch.cat([signal, pad], dim=0)

    if num==9:
        segments, pad_left_len, pad_right_len = fit_segment_and_impute_align_right(signal, time_list, period)
    if num==8:
        segments, pad_left_len, pad_right_len = split_segments_from_shift_with_padding_fast(signal, period, shift)
    
    return segments, pad_left_len, pad_right_len


def process_signal_to_2d_batch(signal: torch.Tensor, time_list: torch.Tensor, period: int, pred_len: int = 24, is_full: bool = True, num=8):
    B, T, C = signal.shape
    batch_segments = []
    pad_left_lens = []
    pad_right_lens = []

    max_segments = 0

    for i in range(B):
        single_signal = signal[i]  # (T, C)
        single_time = time_list[i]
        segments, pad_left_len, pad_right_len = process_signal_to_2d(single_signal, single_time, period, pred_len, is_full)
        batch_segments.append(segments)
        pad_left_lens.append(pad_left_len)
        pad_right_lens.append(pad_right_len)
        max_segments = max(max_segments, segments.shape[0])

    padded_segments = []
    for seg in batch_segments:
        N, P, C = seg.shape
        if N < max_segments:
            pad_n = max_segments - N
            pad_tensor = torch.full((pad_n, P, C), -1.0, dtype=seg.dtype, device=seg.device)
            seg = torch.cat([seg, pad_tensor], dim=0)
        padded_segments.append(seg)

    batch_segments = torch.stack(padded_segments, dim=0)
    return batch_segments, pad_left_lens, pad_right_lens