import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from model import StockTransformer
import transformer_engine.pytorch as te
import transformer_engine.common.recipe as recipe
import random

import os
import shutil

def main():
    # Setup/clean checkpoints directory
    checkpoints_dir = '/home/ienliven/Projects/arcllm/checkpoints'
    if os.path.exists(checkpoints_dir):
        shutil.rmtree(checkpoints_dir)
    os.makedirs(checkpoints_dir)
    
    torch.backends.cudnn.deterministic = False
    
    # 1. Load data and compute log-returns
    df_close = pd.read_csv('/home/ienliven/Projects/arcllm/sp500_close.csv')
    df_volume = pd.read_csv('/home/ienliven/Projects/arcllm/sp500_volume.csv')
    df_macro = pd.read_csv('/home/ienliven/Projects/arcllm/macro_close.csv')

    price_cols = [c for c in df_close.columns if c != 'Date']
    prices = df_close[price_cols].values 
    volumes = df_volume[price_cols].values 
    macro_cols = ['BTC-USD', 'ETH-USD', 'GC=F', 'BZ=F', 'DX-Y.NYB', '^TNX', '^VIX', '^IXIC', '^DJI', '^RUT']
    macro_prices = df_macro[macro_cols].values 

    eps = 1e-8
    prices = np.clip(prices, eps, None)
    volumes = np.clip(volumes, eps, None)
    macro_prices = np.clip(macro_prices, eps, None)

    log_returns = np.log(prices[1:] / prices[:-1]) 
    volume_returns = np.log(volumes[1:] / volumes[:-1])
    macro_returns = np.log(macro_prices[1:] / macro_prices[:-1])

    # Compute Moving Averages and avoid lookahead bias by removing bfill
    df_prices = pd.DataFrame(prices)
    ma20 = df_prices.rolling(window=20).mean().values
    ma50 = df_prices.rolling(window=50).mean().values
    ma200 = df_prices.rolling(window=200).mean().values

    ma20 = np.clip(ma20, eps, None)
    ma50 = np.clip(ma50, eps, None)
    ma200 = np.clip(ma200, eps, None)

    ma20_ratio = np.log(prices / ma20)[1:]
    ma50_ratio = np.log(prices / ma50)[1:]
    ma200_ratio = np.log(prices / ma200)[1:]

    # Compute Moving Averages for macro prices
    df_macro_prices = pd.DataFrame(macro_prices)
    macro_ma20 = df_macro_prices.rolling(window=20).mean().values
    macro_ma50 = df_macro_prices.rolling(window=50).mean().values
    macro_ma200 = df_macro_prices.rolling(window=200).mean().values

    macro_ma20 = np.clip(macro_ma20, eps, None)
    macro_ma50 = np.clip(macro_ma50, eps, None)
    macro_ma200 = np.clip(macro_ma200, eps, None)

    macro_ma20_ratio = np.log(macro_prices / macro_ma20)[1:]
    macro_ma50_ratio = np.log(macro_prices / macro_ma50)[1:]
    macro_ma200_ratio = np.log(macro_prices / macro_ma200)[1:]

    # Filter for the last 4 years (approx 1008 trading days)
    log_returns = log_returns[-1008:] 
    volume_returns = volume_returns[-1008:] 
    macro_returns = macro_returns[-1008:] 
    ma20_ratio = ma20_ratio[-1008:]
    ma50_ratio = ma50_ratio[-1008:]
    ma200_ratio = ma200_ratio[-1008:] 
    macro_ma20_ratio = macro_ma20_ratio[-1008:]
    macro_ma50_ratio = macro_ma50_ratio[-1008:]
    macro_ma200_ratio = macro_ma200_ratio[-1008:] 

    num_days, num_tickers = log_returns.shape
    print(f"Data shape: {num_days} days, {num_tickers} tickers")

    # 2. Build Dataset & Targets
    padded_returns = np.zeros((num_days, 3072), dtype=np.float32)
    padded_returns[:, :num_tickers] = log_returns
    padded_returns[:, 500 : 500 + num_tickers] = volume_returns
    padded_returns[:, 1000 : 1000 + len(macro_cols)] = macro_returns
    padded_returns[:, 1100 : 1100 + num_tickers] = ma20_ratio
    padded_returns[:, 1600 : 1600 + num_tickers] = ma50_ratio
    padded_returns[:, 2100 : 2100 + num_tickers] = ma200_ratio
    padded_returns[:, 2600 : 2600 + len(macro_cols)] = macro_ma20_ratio
    padded_returns[:, 2610 : 2610 + len(macro_cols)] = macro_ma50_ratio
    padded_returns[:, 2620 : 2620 + len(macro_cols)] = macro_ma200_ratio

    # Use a safe maximum dimension for targets to avoid uninitialized data slicing anomalies
    padded_targets = np.zeros((num_days, 512), dtype=np.float32)
    binary_targets = (log_returns > 0).astype(np.float32)
    padded_targets[:-1, :num_tickers] = binary_targets[1:] 

    device = torch.device('cuda')
    X_tensor = torch.tensor(padded_returns, dtype=torch.bfloat16, device=device)
    Y_tensor = torch.tensor(padded_targets, dtype=torch.bfloat16, device=device)

    # 3. Secure Train / Val split chronologically (Past -> Future)
    train_ratio = 0.90
    train_end_idx = int(num_days * train_ratio)
    seq_len = 64 

    train_inputs, train_targets = [], []
    val_inputs, val_targets = [], []
    
    # Train on the first 90%
    for t in range(0, train_end_idx - seq_len):
        train_inputs.append(X_tensor[t : t + seq_len])
        train_targets.append(Y_tensor[t : t + seq_len])
        
    # Validate on the final 10%
    for t in range(train_end_idx, num_days - seq_len):
        val_inputs.append(X_tensor[t : t + seq_len])
        val_targets.append(Y_tensor[t : t + seq_len])
        
    train_inputs = torch.stack(train_inputs) 
    train_targets = torch.stack(train_targets) 
    
    val_inputs = torch.stack(val_inputs)
    val_targets = torch.stack(val_targets)
    
    # Recipe for FP8 Block Scaling on Blackwell GPUs
    r = recipe.DelayedScaling(fp8_format=recipe.Format.E4M3)

    # 4. Ensemble Training Loop
    num_seeds = 3
    seeds = [random.randint(1, 10000) for _ in range(num_seeds)]
    print(f"Training ensemble of {num_seeds} models with seeds: {seeds}")
    
    epochs = 400
    batch_size = 256
    num_samples = train_inputs.size(0)
    
    input_noise_std = 0.01
    gradient_noise_std = 1e-5
    
    mask = torch.zeros(3072, dtype=torch.bfloat16, device=device)
    mask[:num_tickers] = 1.0
    mask[500 : 500 + num_tickers] = 1.0
    mask[1000 : 1000 + len(macro_cols)] = 1.0
    mask[1100 : 1100 + num_tickers] = 1.0
    mask[1600 : 1600 + num_tickers] = 1.0
    mask[2100 : 2100 + num_tickers] = 1.0
    mask[2600 : 2600 + len(macro_cols)] = 1.0
    mask[2610 : 2610 + len(macro_cols)] = 1.0
    mask[2620 : 2620 + len(macro_cols)] = 1.0
    
    for seed_idx, seed in enumerate(seeds, 1):
        print(f"\n--- Training Model {seed_idx}/{num_seeds} (Seed: {seed}) ---")
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        model = StockTransformer(d_feat=3072, seq_len=seq_len).to(device=device, dtype=torch.bfloat16)
        optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)
        loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        
        model.train()
        for epoch in range(1, epochs + 1):
            indices = torch.randperm(num_samples)
            epoch_loss = 0.0
            
            # --- Training Pass ---
            model.train()
            for i in range(0, num_samples, batch_size):
                batch_idx = indices[i : i + batch_size]
                bx = train_inputs[batch_idx].clone()  
                by = train_targets[batch_idx]
                
                if input_noise_std > 0:
                    bx = bx + torch.randn_like(bx) * input_noise_std * mask
                
                optimizer.zero_grad()
                with te.autocast(enabled=True, recipe=r):
                    out = model(bx)
                    raw_loss = loss_fn(out.float(), by.float())
                    loss = raw_loss[:, :, :num_tickers].mean()
                    
                loss.backward()
                
                if gradient_noise_std > 0:
                    for param in model.parameters():
                        if param.grad is not None:
                            noise = torch.randn_like(param.grad) * gradient_noise_std
                            param.grad.add_(noise)
                            
                optimizer.step()
                epoch_loss += loss.item() * len(batch_idx)
                
            epoch_loss /= num_samples
            
            # --- Validation Pass ---
            model.eval()
            with torch.no_grad():
                with te.autocast(enabled=True, recipe=r):
                    val_out = model(val_inputs)
                    val_raw_loss = loss_fn(val_out.float(), val_targets.float())
                    val_loss = val_raw_loss[:, :, :num_tickers].mean().item()
            
            if epoch % 50 == 0 or epoch == 1:
                print(f"Epoch {epoch:03d}/{epochs:03d} | Train BCE: {epoch_loss:.6f} | Val BCE: {val_loss:.6f}")
                
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'num_tickers': num_tickers,
            'seq_len': seq_len,
            'seed': seed
        }
        checkpoint_path = f'{checkpoints_dir}/stock_transformer_seed_{seed}.pt'
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    # 5. Backtest on Val/Test split using the ensemble
    print("\nRunning backtest with the ensemble of models...")
    models = []
    for seed in seeds:
        m = StockTransformer(d_feat=3072, seq_len=seq_len).to(device=device, dtype=torch.bfloat16)
        checkpoint_path = f'{checkpoints_dir}/stock_transformer_seed_{seed}.pt'
        checkpoint = torch.load(checkpoint_path)
        m.load_state_dict(checkpoint['model_state_dict'])
        m.eval()
        models.append(m)
        
    portfolio_capital = 1.0
    benchmark_capital = 1.0
    
    with torch.no_grad():
        # Evaluating across the strict validation slice (chronological future)
        for t in range(train_end_idx + seq_len - 1, num_days - 1):
            seq_x = X_tensor[t - seq_len + 1 : t + 1].unsqueeze(0)
            
            ensemble_probs = np.zeros(num_tickers)
            for m in models:
                with te.autocast(enabled=True, recipe=r):
                    out = m(seq_x)
                logits = out[0, -1, :num_tickers] 
                probs = torch.sigmoid(logits).float().cpu().numpy()
                ensemble_probs += probs
            ensemble_probs /= len(models)
            
            K = 50
            top_k_indices = np.argsort(ensemble_probs)[-K:]
            
            next_day_returns = log_returns[t + 1] 
            portfolio_day_return = np.mean(next_day_returns[top_k_indices])
            
            portfolio_capital *= np.exp(portfolio_day_return)
            
            benchmark_day_return = np.mean(next_day_returns)
            benchmark_capital *= np.exp(benchmark_day_return)

    print("Backtest complete!")
    print(f"Final Ensemble Portfolio Capital: {portfolio_capital:.4f}")
    print(f"Final S&P 500 Benchmark Capital: {benchmark_capital:.4f}")

if __name__ == "__main__":
    main()