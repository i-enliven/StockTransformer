import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from model import StockTransformer
import transformer_engine.pytorch as te
import transformer_engine.common.recipe as recipe

def main():
    # 1. Load data and compute log-returns
    df_close = pd.read_csv('/home/ienliven/Projects/arcllm/sp500_close.csv')
    df_volume = pd.read_csv('/home/ienliven/Projects/arcllm/sp500_volume.csv')
    
    # Exclude Date column
    price_cols = [c for c in df_close.columns if c != 'Date']
    prices = df_close[price_cols].values # [num_days, 500]
    volumes = df_volume[price_cols].values # [num_days, 500]
    
    # Compute log returns: R_t = log(P_t / P_{t-1})
    log_returns = np.log(prices[1:] / prices[:-1])
    
    # Compute log returns of volume: V_t = log(V_t / V_{t-1})
    volume_returns = np.log(np.clip(volumes[1:], 1e-8, None) / np.clip(volumes[:-1], 1e-8, None))
    
    # Filter for the last 2 years (approx 504 trading days)
    # 504 returns require 505 price/volume days
    log_returns = log_returns[-504:] # [504, 500]
    volume_returns = volume_returns[-504:] # [504, 500]
    
    num_days, num_tickers = log_returns.shape
    print(f"Data shape: {num_days} days, {num_tickers} tickers")
    
    # 2. Build Dataset & Targets
    # Input is padded to 1024 dimensions (FP4 requirement)
    padded_returns = np.zeros((num_days, 1024), dtype=np.float32)
    padded_returns[:, :num_tickers] = log_returns
    padded_returns[:, 500 : 500 + num_tickers] = volume_returns
    
    # Targets are binary: 1 if next-day return > 0, else 0
    # Pad to 512 dimensions (FP4 requirement)
    padded_targets = np.zeros((num_days, 512), dtype=np.float32)
    binary_targets = (log_returns > 0).astype(np.float32)
    padded_targets[:-1, :num_tickers] = binary_targets[1:] # target at t is tomorrow's return (t+1)
    
    # Convert to PyTorch tensors
    device = torch.device('cuda')
    X_tensor = torch.tensor(padded_returns, dtype=torch.bfloat16, device=device)
    Y_tensor = torch.tensor(padded_targets, dtype=torch.bfloat16, device=device)
    
    # 3. Create Train / Test split chronologically
    # Train days: 0 to 400. Val/Test days: 400 to 503
    train_end_idx = 400
    seq_len = 64
    
    # Construct train sequences
    train_inputs = []
    train_targets = []
    for t in range(train_end_idx - seq_len):
        train_inputs.append(X_tensor[t : t + seq_len])
        train_targets.append(Y_tensor[t : t + seq_len])
        
    train_inputs = torch.stack(train_inputs)     # [num_sequences, seq_len, 1024]
    train_targets = torch.stack(train_targets)   # [num_sequences, seq_len, 512]
    
    # 4. Instantiate Model, Optimizer, and Loss function
    model = StockTransformer(seq_len=seq_len).to(device=device, dtype=torch.bfloat16)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    
    # Recipe for FP4 Block Scaling
    r = recipe.NVFP4BlockScaling(disable_rht=True)
    
    # 5. Training Loop
    epochs = 2000
    batch_size = 128
    num_samples = train_inputs.size(0)
    
    print("Starting training...")
    model.train()
    for epoch in range(1, epochs + 1):
        # Shuffle batches
        indices = torch.randperm(num_samples)
        epoch_loss = 0.0
        
        for i in range(0, num_samples, batch_size):
            batch_idx = indices[i : i + batch_size]
            bx = train_inputs[batch_idx]
            by = train_targets[batch_idx]
            
            optimizer.zero_grad()
            with te.autocast(enabled=True, recipe=r):
                out = model(bx)
                # Compute loss only on the first 500 active ticker channels
                raw_loss = loss_fn(out, by)
                loss = raw_loss[:, :, :num_tickers].mean()
                
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_idx)
            
        epoch_loss /= num_samples
        if epoch % 100 == 0 or epoch == 1:
            print(f"Epoch {epoch:04d}/{epochs:04d} | Train BCE Loss: {epoch_loss:.6f}")
            
    # Save the model state dict and settings
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'num_tickers': num_tickers,
        'seq_len': seq_len
    }
    torch.save(checkpoint, '/home/ienliven/Projects/arcllm/stock_transformer.pt')
    print("Saved checkpoint to stock_transformer.pt")
    
    # 6. Backtest on Val/Test split
    print("\nRunning backtest...")
    model.eval()
    
    portfolio_capital = 1.0
    benchmark_capital = 1.0
    portfolio_history = [portfolio_capital]
    benchmark_history = [benchmark_capital]
    
    # We step through each day in the val/test set
    with torch.no_grad():
        for t in range(train_end_idx, num_days - 1):
            # Input sequence is the last seq_len days ending at day t
            seq_x = X_tensor[t - seq_len + 1 : t + 1].unsqueeze(0) # [1, seq_len, 1024]
            
            with te.autocast(enabled=True, recipe=r):
                out = model(seq_x) # [1, seq_len, 512]
                
            # Logits for the next day prediction is at the last sequence position
            logits = out[0, -1, :num_tickers] # [500]
            probs = torch.sigmoid(logits).float().cpu().numpy()
            
            # Select top K stocks with the highest buy probability
            K = 10
            top_k_indices = np.argsort(probs)[-K:]
            
            # Next day returns (t+1)
            next_day_returns = log_returns[t + 1] # [500]
            
            # Portfolio return is the average return of the selected top K stocks
            portfolio_day_return = np.mean(next_day_returns[top_k_indices])
            # Convert log return back to simple return multiplier: exp(R)
            portfolio_capital *= np.exp(portfolio_day_return)
            portfolio_history.append(portfolio_capital)
            
            # Benchmark return (equal-weight S&P 500 index)
            benchmark_day_return = np.mean(next_day_returns)
            benchmark_capital *= np.exp(benchmark_day_return)
            benchmark_history.append(benchmark_capital)
            
    print("Backtest complete!")
    print(f"Final Portfolio Capital: {portfolio_capital:.4f}")
    print(f"Final S&P 500 Benchmark Capital: {benchmark_capital:.4f}")

if __name__ == "__main__":
    main()
