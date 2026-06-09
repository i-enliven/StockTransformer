import torch
import pandas as pd
import numpy as np
from model import StockTransformer
import transformer_engine.pytorch as te
import transformer_engine.common.recipe as recipe

def main():
    # 1. Load checkpoint
    checkpoint_path = '/home/ienliven/Projects/arcllm/stock_transformer.pt'
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path)
    
    num_tickers = checkpoint['num_tickers']
    seq_len = checkpoint['seq_len']
    
    # 2. Instantiate and load model
    device = torch.device('cuda')
    model = StockTransformer(seq_len=seq_len).to(device=device, dtype=torch.bfloat16)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 3. Load S&P 500 close prices
    df = pd.read_csv('/home/ienliven/Projects/arcllm/sp500_close.csv')
    price_cols = [c for c in df.columns if c != 'Date']
    ticker_names = price_cols
    
    # Get last seq_len + 1 days of close prices to calculate seq_len returns
    last_prices = df[price_cols].values[-(seq_len + 1):] # [seq_len + 1, 500]
    
    # Compute log-returns
    log_returns = np.log(last_prices[1:] / last_prices[:-1]) # [seq_len, 500]
    
    # Pad input to 512 dimensions (FP4 requirement)
    padded_returns = np.zeros((1, seq_len, 512), dtype=np.float32)
    padded_returns[0, :, :num_tickers] = log_returns
    
    # Convert input to tensor
    X_tensor = torch.tensor(padded_returns, dtype=torch.bfloat16, device=device)
    
    # 4. Model inference
    r = recipe.NVFP4BlockScaling(disable_rht=True)
    with torch.no_grad():
        with te.autocast(enabled=True, recipe=r):
            out = model(X_tensor) # [1, seq_len, 512]
            
    # Extract prediction logits for the last time step
    logits = out[0, -1, :num_tickers] # [500]
    probs = torch.sigmoid(logits).float().cpu().numpy()
    
    # 5. Output top recommendations
    K = 10
    top_k_indices = np.argsort(probs)[-K:][::-1] # sorted in descending order
    
    print("\n" + "="*50)
    print("  STOCK FORECASTING TRANSFORMER BUY RECOMMENDATIONS")
    print("="*50)
    print(f"Top {K} recommended S&P 500 stock tickers to buy for next trading day:")
    for idx, rank in enumerate(top_k_indices, 1):
        ticker = ticker_names[rank]
        prob = probs[rank]
        print(f" Rank {idx:2d}: {ticker:<5} | Buy Probability: {prob:.4%}")
    print("="*50)

if __name__ == "__main__":
    main()
