import torch
import torch.nn as nn
import transformer_engine.pytorch as te
import transformer_engine.common.recipe as recipe

class StockTransformer(nn.Module):
    def __init__(self, num_tickers=500, d_model=256, nhead=8, num_layers=4, ffn_hidden_size=1024, seq_len=64):
        super().__init__()
        self.num_tickers = num_tickers
        self.d_model = d_model
        
        # Input projection
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
        
        # Output projection
        self.output_proj = te.Linear(d_model, 512, bias=True)
        
    def forward(self, x):
        h = self.input_proj(x)
        h = h + self.pos_embed[:, :x.size(1), :]
        for layer in self.layers:
            h = layer(h)
        h = self.ln_f(h)
        out = self.output_proj(h)
        return out

def main():
    # Instantiate model
    model = StockTransformer(seq_len=64).cuda().to(dtype=torch.bfloat16)
    
    # Recipe
    r = recipe.NVFP4BlockScaling(disable_rht=True)
    
    # Input tensor: batch=8, seq_len=64, features=1024
    x = torch.randn(8, 64, 1024, device='cuda', dtype=torch.bfloat16)
    
    # Forward pass
    with te.autocast(enabled=True, recipe=r):
        out = model(x)
        loss = out.sum()
        
    # Backward pass
    loss.backward()
    
    print("SUCCESS: Full model forward/backward pass completed!")
    print("Output shape:", out.shape)

if __name__ == "__main__":
    main()
