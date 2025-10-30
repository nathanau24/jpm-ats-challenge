# jpm-ats-challenge
My solution to JPM ATS German/UK apples challenge using a linear regression model:

This script runs a regression model on the first 80% of the training data, then runs a backtest on the next 20% of data (to avoid data leakage)
The backtest has a very simple entry logic that simply demonstrates that the predictions can lead to profitable results

To run this script, you must rename the csv to 'apples' and add 'Datetime' to the A1 cell of the csv, or just use the apples.csv that is attached in the zip folder

All the output files are already in the repository. You may change the backtest to run from fixed lots -> scaled lots to see a different, more aggressive PnL chart
