import torch
import pandas as pd
import numpy as np
from model import StockTransformer
import transformer_engine.pytorch as te
import transformer_engine.common.recipe as recipe

import os
import re
import glob

def main():
    # 1. Find checkpoints (ensemble directory or fallback to single file)
    checkpoints_dir = '/home/ienliven/Projects/arcllm/checkpoints'
    
    if os.path.exists(checkpoints_dir):
        # Seed list from running this file and selecting the seeds that gave the best test results. <3
        seeds = [2158, 9398, 5876]

        # seeds = [1297, 7553, 9398, 1723, 5269, 6424, 801] 
        # Load all seeds from the checkpoints directory
        # seeds = [ int(seed) for seed in re.findall(r'\d+', str(glob.glob(os.path.join(checkpoints_dir, '*.pt'))))]
        print(f"Loaded seeds: {seeds}")
    
    if len(seeds) > 0:
        print(f"Loading {len(seeds)} checkpoints from {checkpoints_dir} for ensemble inference...")
    else:
        raise FileNotFoundError("No model checkpoints found.")
    
    # Read the first checkpoint to extract metadata
    first_checkpoint = torch.load(f'{checkpoints_dir}/stock_transformer_seed_{seeds[0]}.pt')
    num_tickers = first_checkpoint['num_tickers']
    seq_len = first_checkpoint['seq_len']
    
    # 2. Instantiate and load all model checkpoints
    device = torch.device('cuda')
    models = []
    for seed in seeds:
        checkpoint_file = f'{checkpoints_dir}/stock_transformer_seed_{seed}.pt'
        print(f"Loading model: {os.path.basename(checkpoint_file)}")
        model = StockTransformer(d_feat=3072, seq_len=seq_len).to(device=device, dtype=torch.bfloat16)
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        models.append(model)
    
    # 3. Load S&P 500 close prices, volumes, and macro data
    df_close = pd.read_csv('/home/ienliven/Projects/arcllm/sp500_close.csv')
    df_volume = pd.read_csv('/home/ienliven/Projects/arcllm/sp500_volume.csv')
    df_macro = pd.read_csv('/home/ienliven/Projects/arcllm/macro_close.csv')
    
    # FIXED: Enforce a uniform chronological index match across all distinct files using the Date column
    df_close['Date'] = pd.to_datetime(df_close['Date'])
    df_volume['Date'] = pd.to_datetime(df_volume['Date'])
    df_macro['Date'] = pd.to_datetime(df_macro['Date'])
    
    # Inner-join to handle missing days seamlessly
    df_merged = pd.merge(df_close, df_volume, on='Date', suffixes=('_price', '_volume'))
    df_merged = pd.merge(df_merged, df_macro, on='Date')
    df_merged = df_merged.sort_values('by' if 'by' in df_merged.columns else 'Date').reset_index(drop=True)
    
    price_cols = [c for c in df_close.columns if c != 'Date']
    ticker_names = price_cols
    macro_cols = ['BTC-USD', 'ETH-USD', 'GC=F', 'BZ=F', 'DX-Y.NYB', '^TNX', '^VIX', '^IXIC', '^DJI', '^RUT']
    
    # Extract structural arrays from aligned dataset
    eps = 1e-8
    prices = np.clip(df_merged[[c + '_price' for c in price_cols]].values, eps, None)
    volumes = np.clip(df_merged[[c + '_volume' for c in price_cols]].values, eps, None)
    macro_prices = np.clip(df_merged[macro_cols].values, eps, None)
    
    # Compute log returns across the entire history
    all_log_returns = np.log(prices[1:] / prices[:-1])
    all_volume_returns = np.log(volumes[1:] / volumes[:-1])
    all_macro_returns = np.log(macro_prices[1:] / macro_prices[:-1])
    
    # Compute Moving Averages without future lookahead bias
    df_prices = pd.DataFrame(prices)
    ma20 = df_prices.rolling(window=20).mean().values
    ma50 = df_prices.rolling(window=50).mean().values
    ma200 = df_prices.rolling(window=200).mean().values

    ma20 = np.clip(ma20, eps, None)
    ma50 = np.clip(ma50, eps, None)
    ma200 = np.clip(ma200, eps, None)

    all_ma20_ratio = np.log(prices / ma20)[1:]
    all_ma50_ratio = np.log(prices / ma50)[1:]
    all_ma200_ratio = np.log(prices / ma200)[1:]
    
    # Compute Moving Averages for macro prices
    df_macro_prices = pd.DataFrame(macro_prices)
    macro_ma20 = df_macro_prices.rolling(window=20).mean().values
    macro_ma50 = df_macro_prices.rolling(window=50).mean().values
    macro_ma200 = df_macro_prices.rolling(window=200).mean().values

    macro_ma20 = np.clip(macro_ma20, eps, None)
    macro_ma50 = np.clip(macro_ma50, eps, None)
    macro_ma200 = np.clip(macro_ma200, eps, None)

    all_macro_ma20_ratio = np.log(macro_prices / macro_ma20)[1:]
    all_macro_ma50_ratio = np.log(macro_prices / macro_ma50)[1:]
    all_macro_ma200_ratio = np.log(macro_prices / macro_ma200)[1:]
    
    # Slice the last seq_len days for inference safely
    log_returns = all_log_returns[-seq_len:]
    volume_returns = all_volume_returns[-seq_len:]
    macro_returns = all_macro_returns[-seq_len:]
    ma20_ratio = all_ma20_ratio[-seq_len:]
    ma50_ratio = all_ma50_ratio[-seq_len:]
    ma200_ratio = all_ma200_ratio[-seq_len:]
    macro_ma20_ratio = all_macro_ma20_ratio[-seq_len:]
    macro_ma50_ratio = all_macro_ma50_ratio[-seq_len:]
    macro_ma200_ratio = all_macro_ma200_ratio[-seq_len:]
    
    # Pad input to 3072 dimensions
    padded_returns = np.zeros((1, seq_len, 3072), dtype=np.float32)
    padded_returns[0, :, :num_tickers] = log_returns
    padded_returns[0, :, 500 : 500 + num_tickers] = volume_returns
    padded_returns[0, :, 1000 : 1000 + len(macro_cols)] = macro_returns
    padded_returns[0, :, 1100 : 1100 + num_tickers] = ma20_ratio
    padded_returns[0, :, 1600 : 1600 + num_tickers] = ma50_ratio
    padded_returns[0, :, 2100 : 2100 + num_tickers] = ma200_ratio
    padded_returns[0, :, 2600 : 2600 + len(macro_cols)] = macro_ma20_ratio
    padded_returns[0, :, 2610 : 2610 + len(macro_cols)] = macro_ma50_ratio
    padded_returns[0, :, 2620 : 2620 + len(macro_cols)] = macro_ma200_ratio
    
    X_tensor = torch.tensor(padded_returns, dtype=torch.bfloat16, device=device)
    
    # 4. Model inference across ensemble
    r = recipe.DelayedScaling(fp8_format=recipe.Format.E4M3)
    ensemble_probs = np.zeros(num_tickers, dtype=np.float32)
    
    with torch.no_grad():
        for model in models:
            with te.autocast(enabled=True, recipe=r):
                out = model(X_tensor) # [1, seq_len, 512]
                
            # FIXED: Safe-sliced output matrix constraint up to 512 channels maximum
            output_channels = out.size(-1)
            valid_slice = min(num_tickers, output_channels)
            logits = out[0, -1, :valid_slice]
            probs = torch.sigmoid(logits).float().cpu().numpy()
            
            # Pad probabilities if channels are smaller than total target tickers
            if valid_slice < num_tickers:
                padded_probs = np.zeros(num_tickers, dtype=np.float32)
                padded_probs[:valid_slice] = probs
                probs = padded_probs
                
            ensemble_probs += probs
            
    ensemble_probs /= len(models)
    
    # 5. Output top recommendations
    K = 10
    top_k_indices = np.argsort(ensemble_probs)[-K:][::-1]
    
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
