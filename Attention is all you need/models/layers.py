import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout_ratio, device):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.dropout = nn.Dropout(dropout_ratio)

        # q,k,v에 적용할 FC layer
        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)
        self.fc_o = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]

        def transform_qkv(x):
            # [batch_size, qkv_len, hidden_dim] -> [batch_size, n_heads, qkv_len, head_dim]
            out = self.fc(x) # linear
            out = out.view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)
            return out

        # Q, K, V : [batch_size, n_heads, qkv_len, head_dim]
        Q = self.fc_q(q).view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)
        K = self.fc_k(k).view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)
        V = self.fc_v(v).view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)

        # Scaled-Dot Product Attention
        scale = torch.sqrt(self.hidden_dim) # scale = sqrt(d_k)

        energy = torch.matmul(Q, K.permute(0,1,3,2)) # attention : [batch_size, n_heads, query_len, key_len]

        # 마스크(mask)를 사용하는 경우
        if mask is not None:
            energy = energy.masked_fill(mask==0, -1e10)

        attention = energy / scale
        attention = torch.softmax(self.dropout(attention), dim=-1)

        out = torch.matmul(attention, V) # out : [batch_size, n_heads, query_len, head_dim]
        out = out.permute(0,2,1,3).contiguous() # out : [batch_size, query_len, n_heads, head_dim]
        out = out.view(batch_size, -1, self.hidden_dim) # out : [batch_size, query_len, hidden_dim]
        out = self.fc_o(out) # nn.Linear(self.hidden_dim, self.hidden_dim)

        return out


class PositionWiseFF(nn.Module):
    def __init__(self, hidden_dim, ff_dim, dropout_ratio):
        super().__init__()

        self.fc1 = nn.Linear(hidden_dim, ff_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(ff_dim, hidden_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)

        return out