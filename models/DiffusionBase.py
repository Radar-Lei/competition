import numpy as np
import torch
import torch.nn as nn
from submodules import DataEmbedding, Inception_Block_V1
import torch.fft
import torch.nn.functional as F

def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]

def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=32, dropout=0.2, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k

        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            B, C, num_period, period = out.shape
            # out = out.squeeze(2) # (B, C, L)
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        
        if configs.diff_schedule == "quad":
            self.beta = np.linspace(
                configs.beta_start ** 0.5, configs.beta_end ** 0.5, configs.diff_steps
            ) ** 2
        elif configs.diff_schedule == "linear":
            self.beta = np.linspace(
                configs.beta_start, configs.beta_end, configs.diff_steps
            )
        # self.beta is of shape (T,)
        self.alpha = 1 - self.beta
        """
        self.alpha_hat: cumulative product of alpha, e.g., alpha_hat[i] = alpha[0] * alpha[1] * ... * alpha[i]
        """
        self.alpha_hat = np.cumprod(self.alpha) # self.alpha is still of shape (T,)
        # reshape for computing, self.alpha_torch is of shape (T,) -> (T,1,1)
        self.alpha_torch = torch.tensor(self.alpha_hat).float().to(configs.gpu).unsqueeze(1).unsqueeze(1)

        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.gpu)

        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)        

        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        B,L,K = x_enc.shape

        t = torch.randint(0, self.configs.diff_steps, [B]).to(self.configs.gpu)

        # alpha_torch is of shape (T,1,1), t is of torch.Size([B])
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(x_enc) # (B,L,K)
        noisy_data = (current_alpha ** 0.5) * x_enc + ((1.0 - current_alpha) ** 0.5) * noise

        # mask for the NaN values in the original data
        actual_mask = torch.ne(x_enc, 0).float()
        
        target_mask = actual_mask - mask
        
        cond_obs = mask * x_enc
        noisy_target = (1-mask) * noisy_data
        
        # embedding # enc_out is of shape (B, L_hist, 2*d_model)
        enc_out = self.enc_embedding(cond_obs, noisy_target, x_mark_enc)

        for i in range(self.layer):
            enc_out = self.laryer_norm(self.model[i](enc_out))

        # dec_out is of shape (B, L_pred, K)
        dec_out = self.projection(enc_out)

        return dec_out
