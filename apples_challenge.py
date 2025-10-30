# By: Nathan Au, Egor Khokhlov, Vaibhav Rajesh 
# This script runs a regression model on the first 80% of the training data then runs a backtest on the next 20% of data (to avoid data leakage)
# The backtest has a very simple entry logic that simply demonstrates that the predictions can lead to profitable results

# To run this script, you must rename the csv to 'apples' and add 'Datetime' to the A1 cell of the csv, or just use the apples.csv that is attached in the zip folder

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

df = pd.read_csv('apples.csv', parse_dates=['Datetime'])
df = df.sort_values('Datetime').reset_index(drop=True)

df['german_gbp'] = df['German'] * df['EURGBP']
df['uk_ret'] = df['UK'].pct_change() # model is trained on returns
df['german_ret'] = df['german_gbp'].pct_change()
df = df.dropna().reset_index(drop=True)

split_index = int(len(df) * 0.8) # 80/20 training split
train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy().reset_index(drop=True)

print(f"Training data: {train_df['Datetime'].min().date()} → {train_df['Datetime'].max().date()}")
print(f"Testing data:  {test_df['Datetime'].min().date()} → {test_df['Datetime'].max().date()}")

# train regression model on returns (eg. if German apple prices go up by 1%, how much does the UK price tend to change?)
slope, intercept, r_value, p_value, std_err = stats.linregress(
    train_df['german_ret'], train_df['uk_ret']
)

train_df['uk_pred'] = intercept + slope * train_df['german_ret']
train_df['residual'] = train_df['uk_ret'] - train_df['uk_pred']
print(f"\nModel R² (train): {r_value**2:.4f}")

# apply model to test data
test_df['uk_pred'] = intercept + slope * test_df['german_ret']
test_df['residual'] = test_df['uk_ret'] - test_df['uk_pred']

test_df['uk_pred_price'] = test_df['UK'].iloc[0] * (1 + test_df['uk_pred']).cumprod()
test_df['uk_actual_price'] = test_df['UK']

price_r2 = stats.linregress(test_df['uk_actual_price'], test_df['uk_pred_price']).rvalue ** 2
print(f"Model R² (test, reconstructed prices): {price_r2:.4f}")

sl_thresh = train_df['residual'].std()
upper_thresh = sl_thresh
lower_thresh = -sl_thresh

test_df['signal'] = 0
test_df.loc[test_df['residual'] < lower_thresh, 'signal'] = 1
test_df.loc[test_df['residual'] > upper_thresh, 'signal'] = -1

# backtest - non scaled (fixed lots -> to see good LR)
start_bal = 1000.0
risk_pct = 0.0001 # change the risk_pct to see higher pnl, this was just something I thought was realistic

balance = start_bal
position = 0
lot_size = 0
bal_history = []
pnl_history = []

# use actual UK price change for PnL computation
test_df['uk_chg'] = test_df['UK'].diff()

for i in range(len(test_df)):
    if position != 0:
        daily_pnl = lot_size * test_df.loc[i, 'uk_chg'] * position
        balance += daily_pnl
    else:
        daily_pnl = 0

    if balance <= 0:
        bal_history.extend([0] * (len(test_df) - i))
        pnl_history.extend([0] * (len(test_df) - i))
        print(f"Account wiped out on {test_df.loc[i, 'Datetime'].date()}")
        break

    signal = test_df.loc[i, 'signal']
    
    # entry/exit logic
    if signal == 1 and position != 1:
        position = 1
        lot_size = (start_bal * risk_pct) / sl_thresh  # fixed lots
        # to change to scaled lots, for fun:
        # lot_size = (balance * risk_pct) / sl_thresh

    elif signal == -1 and position != -1:
        position = -1
        lot_size = (start_bal * risk_pct) / sl_thresh 
        # to change to scaled lots, for fun:
        # lot_size = (balance * risk_pct) / sl_thresh

    elif signal == 0 and position != 0:
        position = 0
        lot_size = 0

    bal_history.append(balance)
    pnl_history.append(daily_pnl)

test_df['balance'] = bal_history
test_df['daily_pnl'] = pnl_history

end_bal = test_df['balance'].iloc[-1]

print("Backtest Results (80/20 split):")
print(f"Initial Balance: ${start_bal:.2f}")
print(f"Final Balance:   ${end_bal:.2f}")
print(f"Total Return:    {((end_bal - start_bal) / start_bal) * 100:.2f}%")

trades_pnl = test_df[test_df['daily_pnl'] != 0]['daily_pnl']
sharpe = trades_pnl.mean() / trades_pnl.std() if trades_pnl.std() != 0 else 0
print(f"Sharpe Ratio (non-annualized, on trades): {sharpe:.4f}")

# Plot 1: Actual vs Predicted Price
plt.figure(figsize=(12,6))
plt.plot(test_df['Datetime'], test_df['uk_actual_price'], label='Actual UK Price', color='black')
plt.plot(test_df['Datetime'], test_df['uk_pred_price'], label='Predicted UK Price', color='orange')
plt.xlabel("Date")
plt.ylabel("UK Apple Price (GBP)")
plt.title("UK Apple Price vs Predicted Price (Out-of-Sample)")
plt.legend()
plt.grid(True)
plt.savefig('uk_price_vs_predicted.png') 
plt.close()

# Plot 2: Account Balance
plt.figure(figsize=(12,6))
plt.plot(test_df['Datetime'], test_df['balance'], label='Account Balance (Out-of-Sample)', color='blue')
plt.xlabel("Date")
plt.ylabel("Account Balance (GBP)")
plt.title("Out-of-Sample Account Balance Growth")
plt.legend()
plt.grid(True)
plt.savefig('account_balance_growth.png') 
plt.close() 

test_df.to_csv('apples_with_fixed_thresholds_backtest.csv', index=False)
