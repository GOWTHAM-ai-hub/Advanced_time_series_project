import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerForecaster(nn.Module):
    def __init__(self, n_features, d_model=64, nhead=4, num_layers=3, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.regressor = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        x = self.input_proj(x)
        x = self.pos(x)
        x = self.encoder(x)
        # use last time step representation for prediction
        out = self.regressor(x[:, -1, :])
        return out.squeeze(-1)

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)

    def forward(self, encoder_outputs, hidden):
        # encoder_outputs: (batch, seq_len, hidden_dim)
        # hidden: (batch, hidden_dim) - last hidden state
        seq_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        score = torch.tanh(self.W1(encoder_outputs) + self.W2(hidden))
        attention_weights = torch.softmax(self.V(score), dim=1)  # (batch, seq_len, 1)
        context = torch.sum(attention_weights * encoder_outputs, dim=1)  # (batch, hidden_dim)
        return context, attention_weights

class LSTMWithAttention(nn.Module):
    def __init__(self, n_features, hidden_dim=64, num_layers=1, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.hidden_dim = hidden_dim * (2 if bidirectional else 1)
        self.att = AttentionLayer(self.hidden_dim)
        self.fc = nn.Linear(self.hidden_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        outputs, (h_n, c_n) = self.lstm(x)
        # take last hidden state (concatenate if multi-layer)
        if isinstance(h_n, torch.Tensor):
            last_hidden = h_n[-1]
        else:
            last_hidden = h_n
        context, att_w = self.att(outputs, last_hidden)
        out = self.fc(context)
        return out.squeeze(-1), att_w
