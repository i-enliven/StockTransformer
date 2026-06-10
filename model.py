import torch
import torch.nn as nn
import transformer_engine.pytorch as te

class StockTransformer(nn.Module):
    def __init__(self, d_feat=1024, d_model=256, nhead=4, num_layers=4, ffn_hidden_size=512, seq_len=64, dropout=0.2):
        super().__init__()
        self.d_feat = d_feat
        self.d_model = d_model
        self.seq_len = seq_len
        
        # Linear layer mapping padded features to internal model space
        self.input_proj = te.Linear(d_feat, d_model, bias=True)
        
        # Positional Encoding 
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.dropout = nn.Dropout(p=dropout)
        
        # Transformer Layers
        self.layers = nn.ModuleList([
            te.TransformerLayer(
                hidden_size=d_model,
                ffn_hidden_size=ffn_hidden_size,
                num_attention_heads=nhead,
                self_attn_mask_type='causal',
                # FIXED: Changed from 'bsd' to 'bshd' to satisfy DotProductAttention constraints
                attn_input_format='bshd',
                bias=True,
                hidden_dropout=dropout,
                attention_dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Norm and output layers
        self.ln_f = te.LayerNorm(d_model)
        self.output_proj = te.Linear(d_model, 512, bias=True) # Project back to 512 channel space
        
    def forward(self, x):
        # x expected layout from training loop: [batch_size, seq_len, 1024]
        b, s, f = x.shape
        
        h = self.input_proj(x)
        h = h + self.pos_embed[:, :s, :]
        h = self.dropout(h)
        
        for layer in self.layers:
            h = layer(h)
            
        h = self.ln_f(h)
        out = self.output_proj(h)
        return out
