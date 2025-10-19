import torch
import torch.nn as nn
import torch.nn.functional as F

class LambdaMixer(nn.Module):
    def __init__(self, num_nodes: int, num_branches: int, init_bias_out: float = 0.5):
        super().__init__()
        w = torch.zeros(num_nodes, num_branches)   # (N, K)
        w[:, 0] = init_bias_out
        self.w = nn.Parameter(w)

    def forward(self, inputs):
        num_branches = len(inputs)
        assert num_branches == self.w.shape[1], \
            f"Expected {self.w.shape[1]} branches, got {num_branches}"

        lam = F.softmax(self.w, dim=-1)           # (N, K)
        lam = lam.transpose(0, 1)                 # (K, N)
        lam = lam.view(num_branches, 1, -1, 1, 1) # (K, 1, N, 1, 1)

        stack = torch.stack(inputs, dim=0)        # (K, B, N, P_num, Period)
        mixed = (lam * stack).sum(dim=0)          # (B, N, P_num, Period)

        return mixed, lam.squeeze(-1).squeeze(-1).squeeze(1)  # (B, N, P_num, Period), (K, N)

class ChildRelationMHA(nn.Module):
    def __init__(self, d_model, nhead=6, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model,
                                         num_heads=nhead,
                                         dropout=dropout,
                                         batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        out, attn = self.mha(x, x, x,
                             key_padding_mask=mask,
                             need_weights=True,
                             average_attn_weights=False)
        out = self.norm(x + self.dropout(out))
        attn = attn.mean(dim=1)

        return out

class MuFusionModule(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, samples, point_pred):
        B, F, N, _ = samples.shape
        point_pred_exp = point_pred.unsqueeze(2).expand(-1, -1, N, -1)
        combined = torch.cat([samples, point_pred_exp], dim=-1)  # (B, F, N, 2)
        alpha = torch.sigmoid(self.mlp(combined))                # (B, F, N, 1)
        # alpha = 1.
        fused = alpha * point_pred_exp + (1 - alpha) * samples
        return fused
