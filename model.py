import torch
import torch.nn as nn
import transformer_engine.pytorch as te

class StockTransformer(nn.Module):
    def __init__(self, num_tickers=500, d_model=256, nhead=8, num_layers=4, ffn_hidden_size=1024, seq_len=64):
        super().__init__()
        self.num_tickers = num_tickers
        self.d_model = d_model
        self.seq_len = seq_len
        
        # Input projection (1024 is used because FP4 requires multiples of 16)
        self.input_proj = te.Linear(1024, d_model, bias=True)
        
        # Positional Encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        
        # Transformer Layers
        self.layers = nn.ModuleList([
            te.TransformerLayer(
                hidden_size=d_model,
                ffn_hidden_size=ffn_hidden_size,
                num_attention_heads=nhead,
                self_attn_mask_type='causal',
                attn_input_format='bshd',
                bias=True
            ) for _ in range(num_layers)
        ])
        
        # Layer norm before output projection
        self.ln_f = te.LayerNorm(d_model)
        
        # Output projection (512 features)
        self.output_proj = te.Linear(d_model, 512, bias=True)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, 512]
        h = self.input_proj(x)
        # Add positional embedding
        h = h + self.pos_embed[:, :x.size(1), :]
        for layer in self.layers:
            h = layer(h)
        h = self.ln_f(h)
        out = self.output_proj(h)
        return out
