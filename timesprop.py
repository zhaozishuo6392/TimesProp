import torch
import torch.nn as nn
import torch.nn.functional as F
from Embed import DataEmbedding
import sampling_methods as sampling
import models
from models import ChildRelationMHA
from models import LambdaMixer
import timesprop_tool
from timesprop_tool import PeriodAwareConvBlock


DISTRIBUTION_REGISTRY = {
    "D": {"sampler": sampling.sample_dirichlet,
        "res_dim_fn": lambda C: C},
    "GD": {"sampler": sampling.sample_gd,
        "res_dim_fn": lambda C: 2 * (C - 1)},
    "BL": {"sampler": sampling.sample_bl,
        "res_dim_fn": lambda C: C + 1},
    "SD": {"sampler": sampling.scaled_dirichlet_sampling,
        "res_dim_fn": lambda C: 2 * C},
    "SSD": {"sampler": sampling.sample_ssd_batch,
        "res_dim_fn": lambda C: 2 * C + 1}
}

class PeriodBlock(nn.Module):
    def __init__(self, configs):
        super(PeriodBlock, self).__init__()
        self.mha = configs.mha
        self.period_list = configs.period_list
        self.part_list = configs.part_list
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.period_conv1 = PeriodAwareConvBlock(configs.d_model, configs.d_model)
        self.period_conv2 = nn.ModuleList([
            PeriodAwareConvBlock(configs.d_model, configs.d_model)
            for _ in range(len(self.period_list))
        ])
        self.period_conv3 = nn.ModuleList([
                nn.ModuleList([
                    nn.ModuleList([
                        PeriodAwareConvBlock(configs.d_model, configs.d_model)
                    for ___ in range(self.part_list[_][__])
                    ])
                for __ in range(len(self.part_list[_]))
            ])
            for _ in range(len(self.period_list))
        ])
        self.period_transformer = timesprop_tool.PeriodTransformerEncoder(input_dim=configs.d_model, model_dim=64, num_heads=4, num_layers=2, dropout=0.1)
        
        if self.mha == 1:
            self.MHA_list = nn.ModuleList([ChildRelationMHA(d_model=  (self.seq_len//i+1)*i, nhead=6    ) for i in self.period_list])
            self.MHA_list2 = nn.ModuleList([ChildRelationMHA(d_model=  configs.d_model*i, nhead=4    ) for i in self.period_list])
            self.MHA_list3 = nn.ModuleList([ChildRelationMHA(d_model=  (self.seq_len//i+1)*configs.d_model, nhead=4    ) for i in self.period_list])

        
        if self.mha != 0:
            self.mixer = LambdaMixer(num_nodes=configs.d_model, num_branches=4, init_bias_out=1)
        
    def forward(self, x, x_mark_enc):
        B, T, N = x.size()
        # print(f"==== T={T} ====")
        influence = timesprop_tool.compute_period_influence(x, self.period_list) # shape (B, len(period_list))

        res = []
        for idx, period in enumerate(self.period_list):
            segments, pad_left_lens, pad_right_lens = timesprop_tool.process_signal_to_2d_batch(x, x_mark_enc,
                                                                                          period, self.pred_len, True, 9)
            segments = segments.permute(0, 3, 1, 2).contiguous()  # (B, N, P_num, Period)                
            out1 = self.period_conv2[idx](segments)  # (B, N, P_num, Period)
            
            list_of_out2 = []
            for parts_idx, parts in enumerate(self.part_list[idx]):
                chunks = timesprop_tool.split_tensor_last_dim(out1, parts) # list of (B, N, P_num, Period/part_num)
                for part_idx, part in enumerate(chunks):
                    chunks[part_idx] = self.period_conv3[idx][parts_idx][part_idx](chunks[part_idx])
                out2 = torch.cat(chunks, dim=-1) # (B, N, P_num, Period)
                list_of_out2.append(out2)

            out2_ = sum(list_of_out2)

            out = out1 + out2_
            P_num = out.shape[2]
                        
            Period = out.shape[3]
            if self.mha != 0:

                mha_in1 = out.reshape(B, N, -1)
                mha_in2 = out.permute(0, 2, 1, 3).reshape(B, P_num, -1)
                mha_in3 = out.permute(0, 3, 1, 2).reshape(B, Period, -1)
                
                mha_out = self.MHA_list[idx](mha_in1).reshape(B, N, P_num, -1)
                mha_out2 = self.MHA_list2[idx](mha_in2).reshape(B, P_num, N, -1).permute(0, 2, 1, 3)
                mha_out3 = self.MHA_list3[idx](mha_in3).reshape(B, Period, N, -1).permute(0, 2, 3, 1)
    
                mixed, lam = self.mixer([out, mha_out, mha_out2, mha_out3])
                out = mixed
            

            # ==== End ====
            out = out.permute(0, 2, 3, 1).contiguous()  # → (B, P_num, Period, N)        
            out = out.view(B, -1, out.shape[-1])  # → (B, T_ext=P_num*Period, N)
        
            segments = []
            for b in range(B):
                left = pad_left_lens[b]
                right = pad_right_lens[b]
                right_end = out.shape[1] - right
                if left==0 and right==0:
                    left=24
                segments.append(out[b, left:right_end, :])
            
            segments = torch.stack(segments, dim=0) # (B, seq_len+pred_len, N)
            res.append(segments)
        res = torch.stack(res, dim=-1) # (B, seq_len+pred_len, N, k)
        influence = influence.unsqueeze(1).unsqueeze(1)
        res = res * influence
        res = res.sum(dim=-1) # (B, seq_len+pred_len, N)
        res = res[:, :x.shape[1], :] + x
        return res

class TemporalEncoderShared(nn.Module):
    def __init__(self, hidden_dim, num_nodes, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes

        self.grus = nn.ModuleList([
            nn.GRU(
                input_size=1,
                hidden_size=hidden_dim // 2,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
            for _ in range(num_nodes)
        ])

        self.linear_res = nn.ModuleList([
            nn.Linear(1, hidden_dim) for _ in range(num_nodes)
        ])

        self.gate = nn.ModuleList([
            nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(num_nodes)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_nodes)
        ])

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):  # x: (B, T, N)
        B, T, N = x.shape
        x = x.permute(0, 2, 1)  # (B, N, T)
        h_list = []

        for i in range(N):
            xi = x[:, i, :].unsqueeze(-1)  # (B, T, 1)

            out, _ = self.grus[i](xi)      # (B, T, H)
            h_last = out[:, -1, :]         # (B, H)

            res = self.linear_res[i](xi[:, -1, :])  # (B, H)

            gate_input = torch.cat([h_last, res], dim=-1)  # (B, 2H)
            gate = torch.sigmoid(self.gate[i](gate_input)) # (B, H)
            h = gate * h_last + (1 - gate) * res

            h = self.layer_norms[i](h)
            h = self.activation(h)
            h = self.dropout(h)

            h_list.append(h)

        return torch.stack(h_list, dim=1)  # (B, N, H)


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.C = configs.c_out - 1
        self.total_nodes = self.C + 1

        dist_info = DISTRIBUTION_REGISTRY[configs.dist_type]
        self.sampler = dist_info["sampler"]
        self.res_dim = dist_info["res_dim_fn"](self.C)
        
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        self.node_models = nn.ModuleList([PeriodBlock(configs) for _ in range(1)])

        self.layer_norm = nn.LayerNorm(self.d_model)
        
        self.project_mu = nn.Sequential(
            nn.Linear(self.d_model, configs.c_out)
        )
        
        self.project_sigma = nn.Sequential(
            nn.Linear(self.d_model, self.d_model*2),
            nn.ReLU(),
            nn.Linear(self.d_model*2, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 1)
        )
        self.project_ssd = nn.Sequential(
            nn.Linear(self.C * 2 + self.d_model, self.res_dim)
        )


        self.mu_last_proj = models.MuFusionModule(hidden_dim=8)


        self.lambda_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.C),
            nn.Sigmoid()
        )
        
        self.eps = 1e-6
        
        hidden_dim = self.d_model
        
        self.encoder = TemporalEncoderShared(hidden_dim = hidden_dim, num_nodes=self.C+1)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.seq_len)
        )

        self.decoder2 = nn.Sequential(
            nn.Linear(self.C+1, self.d_model),
            nn.ReLU(),
            nn.LayerNorm(self.d_model)
        )

    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        B, H, _ = x_enc.shape  # (B, H, C+1)

        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        sloth_out = self.encoder(x_enc) # (B, N, H)
        sloth_out = self.decoder(sloth_out.reshape(-1, self.d_model))
        sloth_out = sloth_out.view(B, -1, H).permute(0, 2, 1)  # (B, F, N)
        sloth_out = self.decoder2(sloth_out)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # (B, H, d_model)

        enc_out = sloth_out + enc_out

        
        enc_out = self.layer_norm(self.node_models[0](enc_out, x_mark_enc))  # (B, H, d_model)

        lam_full = self.lambda_head(enc_out)             # (B, H+F, 1)
        lam = lam_full[:, -self.pred_len:, :]            # (B, F, 1)
        lam_s = lam.unsqueeze(2)                         # (B, F, 1, 1)

        mu = self.project_mu(enc_out)         # (B, H+F, C+1)
        sigma = F.softplus(self.project_sigma(enc_out))  # (B, H+F, 1)

        children_raw = mu[..., 1:]  # (B, H+F, C)
        children_percent = torch.clamp(children_raw, min=0.0)
        children_percent = children_percent / (children_percent.sum(dim=-1, keepdim=True) + 1e-4)

        ssd_input = torch.cat([
            children_percent,
            children_raw,
            enc_out
        ], dim=-1)
        dist = self.project_ssd(ssd_input)
        dist = F.softplus(dist)
        dist_middle = dist[..., self.C:2 * self.C]
        dist[..., self.C:2 * self.C] = torch.softmax(dist_middle, dim=-1)
        dist = dist[:, -self.pred_len:, :]  # (B, F, 2C+1)

        mu = mu * stdev[:, 0:1, :] + means[:, 0:1, :]  # (B, H+F, C+1)
        sigma = sigma * stdev[:, 0:1, :]                  # (B, H+F, 1)

        prob = torch.stack([mu[..., 0], sigma[..., 0]], dim=-1)  # (B, H+F, 2)
        prob = prob[:, -self.pred_len:, :]  # (B, F, 2)

        parent_samples = sampling.generate_normal_samples(prob)       # (B, F, N, 1)
        children_samples = self.sampler(dist)            # (B, F, N, C)
        
        parent_samples = self.mu_last_proj(parent_samples, mu[:, -self.pred_len:, :1])
        
        children_samples = parent_samples * children_samples
        
        result_samples = torch.cat([parent_samples, children_samples], dim=-1)  # (B, F, N, C+1)

        samples_out = result_samples
        dec_out = mu[:, -self.pred_len:, :]  # (B, F, C+1)


        return prob, dist, dec_out, samples_out