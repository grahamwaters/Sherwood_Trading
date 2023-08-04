 # Cast to floats and add conditions
signals_df['buy_signal'] = (
    (float(df['ema'])) > float(df['sma']) &
    (float(df['rsi']) < 30) &
    (float(df['williams']) < -80) &
    (float(df['close']) < float(df['bollinger_lower_band'])) &
    (float(df['close']) < float(df['sma'])) &
    (float(df['close']) < float(df['ema'])) &
    (float(float(macd_line)) > float(macd_signal))
)
# Cast to floats and add conditions
signals_df['sell_signal'] = (
    (float(df['ema'])) < float(df['sma']) &
    (float(df['rsi']) > 70) &
    (float(df['williams']) > -20) &
    (float(df['close']) > float(df['bollinger_upper_band'])) &
    (float(df['close']) > float(df['sma'])) &
    (float(df['close']) > float(df['ema'])) &
    (float(float(macd_line)) < float(macd_signal))
)
# Drop the NaNs from the DataFrame
signals_df = signals_df.dropna()
# Print the DataFrame
