import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from submodules import DataEmbedding
from submodules import Inception_Block_V1


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


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.kernel_factor = configs.kernel_factor

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

class SpatialEncoder(nn.Module):
    def __init__(self, channels, heads, layers, t_ff):
        super(SpatialEncoder, self).__init__()
        self.layer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=channels, 
                                                                    nhead=heads, 
                                                                    dim_feedforward=t_ff, 
                                                                    dropout=0.1, 
                                                                    activation="gelu",
                                                                    batch_first=True
                                                                    ), 
                                                                    num_layers=layers,
                                                                    )
    def forward(self, x):
        # x is of shape (B, L, K, d_model)
        B, L, K, d_model = x.shape
        x = x.reshape(B*L, K, d_model)
        x = self.layer(x)
        x = x.reshape(B, L, K, d_model)
        return x
        
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len

        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])

        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # self.fea_transformer = get_torch_trans(heads=2, layers=1, channels=configs.seq_len)
        # self.temporal_transformer = get_torch_trans(heads=2, layers=1, channels=configs.d_model)
        self.spatial_encoders = nn.ModuleList([SpatialEncoder(configs.d_model, configs.nheads, configs.trans_layers, configs.t_ff)
                                               for _ in range(configs.e_layers)])

        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        if self.task_name == 'prediction':
            self.predict_linear = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(
                configs.d_model, 1, bias=True)
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                configs.d_model, 1, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding x_enc is of shape (B, L_hist, K) # T is L_hist
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # enc_out is of shape (B, L_hist, K, d_model)
        # align temporal dimension # [B,L_hist, K, d_model] -> [B,L_hist+h_pred,K,d_model]
        enc_out = self.predict_linear(enc_out.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        B, L, K, d_model = enc_out.shape

        # TimesNet
        for i in range(self.layer):
            enc_out = self.spatial_encoders[i](enc_out) # [B,L,K,d_model]
            enc_out = enc_out.permute(0,2,1,3).reshape(B*K, L, d_model) # [B*K,L,d_model]
            enc_out = self.layer_norm(self.model[i](enc_out)) # [B*K,L,d_model]
            enc_out = enc_out.reshape(B, K, L, d_model).permute(0,2,1,3) # [B,L,K,d_model]

        # porject back [B,L_hist+h_pred,K, d_model] -> [B,L_hist+h_pred,K,1] -> [B,L_hist+h_pred,K]
        dec_out = self.projection(enc_out).squeeze(-1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # (B,L,K,d_model)
        B,L,K,d_model = enc_out.shape
        # TimesNet
        for i in range(self.layer):
            enc_out = self.spatial_encoders[i](enc_out) # [B,L,K,d_model]
            enc_out = enc_out.permute(0,2,1,3).reshape(B*K, L, d_model) # [B*K,L,d_model]
            enc_out = self.layer_norm(self.model[i](enc_out))
            enc_out = enc_out.reshape(B, K, L, d_model).permute(0,2,1,3) # [B,L,K,d_model]
        # porject to (B,L,K,1) -> (B,L,K)
        dec_out = self.projection(enc_out).squeeze(-1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'prediction':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        return None    