import os
import pandas as pd
import yfinance as yf
import numpy as np

def main():
    base_path = '/home/ienliven/Projects/arcllm'
    close_path = os.path.join(base_path, 'sp500_close.csv')
    volume_path = os.path.join(base_path, 'sp500_volume.csv')
    
    # 1. Read existing sp500_close.csv to extract the list of tickers in correct order
    if not os.path.exists(close_path):
        raise FileNotFoundError(f"Original S&P 500 close price CSV not found at: {close_path}")
        
    df_old = pd.read_csv(close_path)
    tickers = [c for c in df_old.columns if c != 'Date']
    
    dummy_tickers = [t for t in tickers if t.startswith('DUMMY')]
    real_tickers = [t for t in tickers if not t.startswith('DUMMY')]
    
    print(f"Loaded {len(tickers)} tickers from existing file ({len(real_tickers)} real, {len(dummy_tickers)} dummy).")
    
    # 2. Download history from Yahoo Finance
    start_date = "2020-01-01"
    print(f"Downloading history from Yahoo Finance starting from {start_date}...")
    
    # yfinance download
    raw_data = yf.download(real_tickers, start=start_date, group_by='column')
    
    print("Download finished. Aligning data...")
    
    # In yfinance, if we download multiple tickers, columns are MultiIndex: (Price, Ticker)
    close_df = raw_data['Close']
    volume_df = raw_data['Volume']
    
    # Convert index (DatetimeIndex) to string Date 'YYYY-MM-DD'
    dates = close_df.index.strftime('%Y-%m-%d')
    
    # Build final close and volume dataframes with the exact order of tickers
    final_close = pd.DataFrame(index=close_df.index)
    final_volume = pd.DataFrame(index=volume_df.index)
    
    for t in tickers:
        if t in real_tickers:
            # Check if ticker is in downloaded dataframe
            if t in close_df.columns:
                final_close[t] = close_df[t]
            else:
                print(f"Warning: Close price for ticker {t} not found in downloaded data. Filling with NaN.")
                final_close[t] = np.nan
                
            if t in volume_df.columns:
                final_volume[t] = volume_df[t]
            else:
                print(f"Warning: Volume for ticker {t} not found in downloaded data. Filling with NaN.")
                final_volume[t] = np.nan
        else:
            # Dummy tickers are filled with 1.0
            final_close[t] = 1.0
            final_volume[t] = 1.0
            
    # Forward fill then backward fill missing data
    final_close = final_close.ffill().bfill()
    final_volume = final_volume.ffill().bfill()
    
    # Fill any remaining NaNs (e.g. if download failed completely for a ticker) with 1.0
    final_close = final_close.fillna(1.0)
    final_volume = final_volume.fillna(1.0)
    
    # Reset index to insert Date column as first column
    final_close.insert(0, 'Date', dates)
    final_volume.insert(0, 'Date', dates)
    
    # Save to CSV
    final_close.to_csv(close_path, index=False)
    final_volume.to_csv(volume_path, index=False)
    
    print(f"Successfully saved close prices to {close_path} (shape: {final_close.shape})")
    print(f"Successfully saved volume to {volume_path} (shape: {final_volume.shape})")
    
    # 3. Download Macro/Crypto Data
    print("\nDownloading macro/crypto assets history from Yahoo Finance...")
    macro_symbols = ['BTC-USD', 'ETH-USD', 'GC=F', 'BZ=F', 'DX-Y.NYB', '^TNX', '^VIX', '^IXIC', '^DJI', '^RUT']
    raw_macro = yf.download(macro_symbols, start=start_date, group_by='column')
    
    # Get Close prices dataframe
    if 'Close' in raw_macro.columns:
        macro_close = raw_macro['Close']
    else:
        # In case only one symbol was requested or returned as a flat index
        macro_close = raw_macro
        
    macro_final = pd.DataFrame(index=close_df.index)
    for sym in macro_symbols:
        if sym in macro_close.columns:
            macro_final[sym] = macro_close[sym]
        else:
            print(f"Warning: Macro symbol {sym} not found in downloaded data. Filling with NaN.")
            macro_final[sym] = np.nan
            
    # Forward fill then backward fill missing data
    macro_final = macro_final.ffill().bfill()
    # Fill any remaining NaNs with 1.0 (to prevent log issues)
    macro_final = macro_final.fillna(1.0)
    
    # Reset index and insert Date column as string
    macro_final = macro_final.reset_index()
    macro_final['Date'] = macro_final['Date'].dt.strftime('%Y-%m-%d')
    macro_final = macro_final[['Date'] + macro_symbols]
    
    # Save macro to CSV
    macro_path = os.path.join(base_path, 'macro_close.csv')
    macro_final.to_csv(macro_path, index=False)
    print(f"Successfully saved macro close prices to {macro_path} (shape: {macro_final.shape})")

if __name__ == "__main__":
    main()
