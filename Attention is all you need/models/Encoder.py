import torch
import torch.nn as nn
from layers import MultiHeadAttention, PositionWiseFF

class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, ff_dim, dropout_ratio, device):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(hidden_dim, n_heads, dropout_ratio)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.position_wise_FF = PositionWiseFF(hidden_dim, ff_dim, dropout_ratio)
        self.FF_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, src, src_mask):
        # src: [batch_size, src_len, hidden_dim]
        # src_mask: [batch_size, src_len]

        attention = self.self_attention(src, src, src, src_mask)
        norm = self.attention_norm(src + self.dropout(attention))
        forward = self.position_wise_FF(norm)
        out = self.FF_norm(norm + self.dropout(forward))

        return out


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, n_heads, ff_dim, dropout_ratio, device, max_length=100):
        super().__init__()
        self.device = device

        self.tok_embedding = nn.Embedding(input_dim, hidden_dim)
        self.pos_embedding = nn.Embedding(max_length, hidden_dim)

        self.layers = nn.ModuleList([EncoderLayer(hidden_dim, n_heads, ff_dim, dropout_ratio, device) for n in range(n_layers)])
        self.dropout = nn.Dropout(dropout_ratio)
        self.scale = torch.sqrt(hidden_dim)

    def forward(self, src, src_mask):
        # src: [batch_size, src_len]
        # src_mask: [batch_size, src_len]

        batch_size = src.shape[0]
        src_len = src.shape[1]

        # positional embedding
        # pos : [batch_size, src_len]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1)

        # src : [batch_size, src_len, hidden_dim]
        src = self.dropout(self.tok_embedding(src) * self.scale + self.pos_embedding(src))

        # Encoder Layer 반복, 순전파
        for layer in self.layers:
            src = layer(src, src_mask)
        
        return src
