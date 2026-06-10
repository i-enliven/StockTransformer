import torch
import pandas as pd
import numpy as np
from model import StockTransformer
import transformer_engine.pytorch as te
import transformer_engine.common.recipe as recipe

import os
import glob

def main():
    # 1. Find checkpoints (ensemble directory or fallback to single file)
    checkpoints_dir = '/home/ienliven/Projects/arcllm/checkpoints'
    fallback_path = '/home/ienliven/Projects/arcllm/stock_transformer.pt'
    
    checkpoint_files = []
    if os.path.exists(checkpoints_dir):
        checkpoint_files = glob.glob(os.path.join(checkpoints_dir, '*.pt'))
        
    if len(checkpoint_files) > 0:
        print(f"Loading {len(checkpoint_files)} checkpoints from {checkpoints_dir} for ensemble inference...")
    elif os.path.exists(fallback_path):
        print(f"No ensemble checkpoints found. Loading single checkpoint from {fallback_path}...")
        checkpoint_files = [fallback_path]
    else:
        raise FileNotFoundError("No model checkpoints found.")
    
    # Read the first checkpoint to extract metadata
    first_checkpoint = torch.load(checkpoint_files[0])
    num_tickers = first_checkpoint['num_tickers']
    seq_len = first_checkpoint['seq_len']
    
    # 2. Instantiate and load all model checkpoints
    device = torch.device('cuda')
    models = []
    for cf in checkpoint_files:
        print(f"Loading model: {os.path.basename(cf)}")
        model = StockTransformer(seq_len=seq_len).to(device=device, dtype=torch.bfloat16)
        checkpoint = torch.load(cf) if cf != checkpoint_files[0] else first_checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        models.append(model)
    
    # 3. Load S&P 500 close prices, volumes, and macro data
    df_close = pd.read_csv('/home/ienliven/Projects/arcllm/sp500_close.csv')
    df_volume = pd.read_csv('/home/ienliven/Projects/arcllm/sp500_volume.csv')
    df_macro = pd.read_csv('/home/ienliven/Projects/arcllm/macro_close.csv')
    price_cols = [c for c in df_close.columns if c != 'Date']
    ticker_names = price_cols
    macro_cols = ['BTC-USD', 'ETH-USD', 'GC=F', 'BZ=F', 'DX-Y.NYB', '^TNX', '^VIX']
    
    # Get last seq_len + 1 days of close prices, volumes, and macro to calculate seq_len returns
    last_prices = df_close[price_cols].values[-(seq_len + 1):] # [seq_len + 1, 500]
    last_volumes = df_volume[price_cols].values[-(seq_len + 1):] # [seq_len + 1, 500]
    last_macro = df_macro[macro_cols].values[-(seq_len + 1):] # [seq_len + 1, 7]
    
    # Compute log-returns of prices, volumes, and macro separately
    log_returns = np.log(last_prices[1:] / last_prices[:-1]) # [seq_len, 500]
    volume_returns = np.log(np.clip(last_volumes[1:], 1e-8, None) / np.clip(last_volumes[:-1], 1e-8, None)) # [seq_len, 500]
    macro_returns = np.log(last_macro[1:] / last_macro[:-1]) # [seq_len, 7]
    
    # Pad input to 1024 dimensions (FP4 requirement)
    padded_returns = np.zeros((1, seq_len, 1024), dtype=np.float32)
    padded_returns[0, :, :num_tickers] = log_returns
    padded_returns[0, :, 500 : 500 + num_tickers] = volume_returns
    padded_returns[0, :, 1000 : 1000 + len(macro_cols)] = macro_returns
    
    # Convert input to tensor
    X_tensor = torch.tensor(padded_returns, dtype=torch.bfloat16, device=device)
    
    # 4. Model inference across ensemble
    r = recipe.NVFP4BlockScaling(disable_rht=True)
    ensemble_probs = np.zeros(num_tickers, dtype=np.float32)
    
    with torch.no_grad():
        for model in models:
            with te.autocast(enabled=True, recipe=r):
                out = model(X_tensor) # [1, seq_len, 512]
                
            # Extract prediction logits for the last time step
            logits = out[0, -1, :num_tickers] # [500]
            probs = torch.sigmoid(logits).float().cpu().numpy()
            ensemble_probs += probs
            
    # Average probabilities
    ensemble_probs /= len(models)
    
    # 5. Output top recommendations
    K = 10
    top_k_indices = np.argsort(ensemble_probs)[-K:][::-1] # sorted in descending order
    
    print("\n" + "="*50)
    print("  STOCK FORECASTING TRANSFORMER ENSEMBLE BUY RECOMMENDATIONS")
    print("="*50)
    print(f"Top {K} recommended S&P 500 stock tickers to buy for next trading day:")
    for idx, rank in enumerate(top_k_indices, 1):
        ticker = ticker_names[rank]
        prob = ensemble_probs[rank]
        print(f" Rank {idx:2d}: {ticker:<5} | Buy Probability: {prob:.4%}")
    print("="*50)
    
    # 6. Save all recommendations to CSV
    predictions_df = pd.DataFrame({
        'Ticker': ticker_names,
        'Buy_Probability': ensemble_probs
    }).sort_values(by='Buy_Probability', ascending=False)
    
    predictions_csv_path = '/home/ienliven/Projects/arcllm/predictions.csv'
    predictions_df.to_csv(predictions_csv_path, index=False)
    print(f"Saved all ticker probabilities to {predictions_csv_path}")

if __name__ == "__main__":
    main()
