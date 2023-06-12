# SherwoodAI
A cryptocurrency trader on Robin Stocks.

We've put together a comprehensive scalping strategy for the trading bot. Here's a summary of the strategy:

1. **Technical Analysis**: the bot will use multiple indicators (moving average, RSI, Bollinger Bands, MACD, and simple moving average) on either a 1-minute or 5-minute price chart.

2. **Signal Strength Calculation**:
   - Buy Signal Strength: Increment by 1 when the price touches the lower Bollinger Band and the RSI is less than 30.
   - Sell Signal Strength: Increment by 1 when the price touches the upper Bollinger Band and the RSI is more than 70.

3. **Stop Losses**: Initial stop loss is set at 1% below the purchase price. A trailing stop loss (TSL) is set at 1% below the current market price and can only increase, not decrease.

4. **Trailing Stop Loss Update**:
   - TSL_NOW is calculated as the current price minus 1% of the purchase price.
   - If TSL_NEW > TSL_PREVIOUS, then TSL_NOW = TSL_NEW.
   - Otherwise, TSL_NOW remains the same as TSL_PREVIOUS.

This appears to be a solid strategy. However, keep in mind a few things:

- the strategy relies heavily on technical indicators, which aren't always perfect. It might be worth considering additional or alternative indicators based on the backtesting results.
- the strategy is designed to be used on a 1-minute or 5-minute price chart. It might be worth considering other timeframes as well.
