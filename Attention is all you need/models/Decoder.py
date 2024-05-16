import torch
import torch.nn as nn
from .layers import MultiHeadAttention, PositionWiseFF

class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, ff_dim, dropout_ratio, device):
        super().__init__()

        self.self_attention = MultiHeadAttention(hidden_dim, n_heads, dropout_ratio)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.encoder_attention = MultiHeadAttention(hidden_dim, n_heads, dropout_ratio)
        self.encoder_norm = nn.LayerNorm(hidden_dim)
        self.position_wise_FF = PositionWiseFF(hidden_dim, ff_dim, dropout_ratio)
        self.FF_norm = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg: [batch_size, trg_len, hidden_dim]
        # enc_src: [batch_size, src_len, hidden_dim]
        # trg_mask: [batch_size, trg_len]
        # src_mask: [batch_size, src_len]

        # self attention
        attention = self.self_attention(trg, trg, trg, trg_mask)
        norm = self.attention_norm(trg + self.dropout(attention))

        attention = self.encoder_attention(norm, enc_src, enc_src, src_mask)
        norm = self.encoder_norm(norm + self.dropout(attention))

        ff = self.position_wise_FF(norm)
        trg = self.FF_norm(norm + self.dropout(ff))

        return trg

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, n_layers, n_heads, ff_dim, dropout_ratio, device, max_length=100):
        super().__init__()

        self.tok_embedding = nn.Embedding(output_dim, hidden_dim)
        self.pos_embedding = nn.Embedding(max_length, hidden_dim)

        self.layers = nn.ModuleList([DecoderLayer(hidden_dim, n_heads, ff_dim, dropout_ratio, device) for n in n_layers])
        self.fc_o = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout_ratio)
        self.scale = torch.sqrt(hidden_dim)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg: [batch_size, trg_len, hidden_dim]
        # enc_src: [batch_size, src_len, hidden_dim]
        # trg_mask: [batch_size, trg_len]
        # src_mask: [batch_size, src_len]

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size) # pos: [batch_size, trg_len]

        # trg: [batch_size, trg_len, hidden_dim]
        trg = self.dropout(self.tok_embedding(trg) * self.scale + self.pos_embedding(pos))

        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)

        out = self.fc_o(trg)

        return out