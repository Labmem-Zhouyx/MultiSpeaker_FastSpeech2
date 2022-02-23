import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.tools import get_mask_from_lengths

class SpeakerEncoder(nn.Module):
    def __init__(self):
        super(SpeakerEncoder, self).__init__()
        self.frameencoder = NormalEncoder()
        self.dsencoder = DownsampleEncoder()
    
    def forward(self, inputs, input_lens):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        max_len = inputs.size(1) // 16
        out_lens = input_lens // 16
        out_masks = (1 - get_mask_from_lengths(out_lens, max_len).float()).unsqueeze(-1).expand(-1, -1, 256).to(device)   # [B, T_y]
        outs = self.frameencoder(inputs)
        outs = self.dsencoder(outs)
        spkemb = torch.sum(outs * out_masks, axis=1) / out_lens.unsqueeze(-1).expand(-1, 256)
 
        return spkemb


class NormalEncoder(nn.Module):
    def __init__(self, in_dim=80, conv_channels=[512, 512], kernel_size=5, stride=1, padding=2, dropout=0.2, out_dim=256):
        super(NormalEncoder, self).__init__()

        # convolution layers followed by batch normalization and ReLU activation
        K = len(conv_channels)

        # 1-D convolution layers
        filters = [in_dim] + conv_channels

        self.conv1ds = nn.ModuleList(
            [nn.Conv1d(in_channels=filters[i],
                       out_channels=filters[i+1],
                       kernel_size=kernel_size,
                       stride=stride,
                       padding=padding)
             for i in range(K)])

        # 1-D batch normalization (BN) layers
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(num_features=conv_channels[i])
             for i in range(K)])

        # ReLU
        self.relu = nn.ReLU()
        
        # dropout
        self.dropout = nn.Dropout(dropout)

        self.outlayer = nn.Linear(in_features=conv_channels[-1], out_features=out_dim)

    def forward(self, x):
        # transpose to (B, embed_dim, T) for convolution, and then back
        out = x.transpose(1, 2)
        for conv, bn in zip(self.conv1ds, self.bns):
            out = conv(out)
            out = self.relu(out)
            out = bn(out)  # [B, 128, T//2^K, mel_dim//2^K], where 128 = conv_channels[-1]
            out = self.dropout(out)

        out = out.transpose(1, 2)  # [B, T//2^K, 128, mel_dim//2^K]
        B, T = out.size(0), out.size(1)
        out = out.contiguous().view(B, T, -1)  # [B, T//2^K, 128*mel_dim//2^K]

        out = self.outlayer(out) 
        return out


class DownsampleEncoder(nn.Module):
    def __init__(self, in_dim=256, conv_channels=[128, 256, 512, 512], kernel_size=3, stride=1, padding=1, dropout=0.2, pooling_sizes=[2, 2, 2, 2], out_dim=256):
        super(DownsampleEncoder, self).__init__()

        K = len(conv_channels)

        # 1-D convolution layers
        filters = [in_dim] + conv_channels
        self.conv1ds = nn.ModuleList(
            [nn.Conv1d(in_channels=filters[i],
                       out_channels=filters[i+1],
                       kernel_size=kernel_size,
                       stride=stride,
                       padding=padding)
             for i in range(K)])

        # 1-D batch normalization (BN) layers
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(num_features=conv_channels[i])
             for i in range(K)])

        self.pools = nn.ModuleList(
            [nn.AvgPool1d(kernel_size=pooling_sizes[i]) for i in range(K)]
        )

        # ReLU
        self.relu = nn.ReLU()

        # dropout
        self.dropout = nn.Dropout(dropout)



        self.local_outlayer = nn.Sequential(
            nn.Linear(in_features=conv_channels[-1],
                      out_features=out_dim),
            nn.Tanh()
        )

    def forward(self, inputs):
        out = inputs.transpose(1, 2)
        for conv, bn, pool in zip(self.conv1ds, self.bns, self.pools):
            out = conv(out)
            out = self.relu(out)
            out = bn(out) # [B, 128, T//2^K, mel_dim//2^K], where 128 = conv_channels[-1]
            out = self.dropout(out)
            out = pool(out)

        out = out.transpose(1, 2)  # [B, T//2^K, 128, mel_dim//2^K]
        B, T = out.size(0), out.size(1)
        out = out.contiguous().view(B, T, -1)  # [B, T//2^K, 128*mel_dim//2^K]

        local_output = self.local_outlayer(out)
        return local_output
