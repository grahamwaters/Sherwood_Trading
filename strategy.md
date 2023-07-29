import pandas_ta

Variables I have access to in the `ta` or `pandas_ta` library:
- `AnalysisIndicators` class (this is the class that contains all the indicators, and is the class that is inherited by the `DataFrame` class)
- `Strategy` - this is the class that contains all the strategies, and is the class that is inherited by the `DataFrame` class. Example: `df.ta.strategy(Strategy)` or `df.ta.strategy(Strategy, *args, **kwargs)`
- `bbands` - this is how we can get the bollinger bands indicator. Example: `df.ta.bbands()` or `df.ta.bbands(close='close', length=20, std=2)` or `df.ta.bbands(close='close', length=20, std=2, append=True)`.
- `ema` - this is how we can get the exponential moving average indicator. Example: `df.ta.ema()` or `df.ta.ema(close='close', length=20)`. We can determine good buys and sells for the bot by checking if the price is above or below the EMA. If the price is above the EMA, it's a good buy. If the price is below the EMA, it's a good sell.
- `rsi` - this is how we can get the relative strength index indicator. Example: `df.ta.rsi()` or `df.ta.rsi(close='close', length=14)`. We can determine good buys and sells for the bot by checking if the RSI is above or below 30 or 70. If the RSI is below 30, it's a good buy. If the RSI is above 70, it's a good sell.
- `stoch` - this is how we can get the stochastic oscillator indicator. Example: `df.ta.stoch()` or `df.ta.stoch(high='high', low='low', close='close')`. We can determine good buys and sells for the bot by checking if the stochastic oscillator is above or below 20 or 80. If the stochastic oscillator is below 20, it's a good buy. If the stochastic oscillator is above 80, it's a good sell.

The most useful indicators for short-term high-frequency day-trading scalping cryptocurrencies are:
    1. EMA - this is a trend indicator, we can use it to determine if the price is going up or down. If the price is above the EMA, it's a good buy. If the price is below the EMA, it's a good sell. Example: `df.ta.ema()` or `df.ta.ema(close='close', length=20)`
    Triggers for EMA:
        - If the price is above the EMA, it's a good buy
        - If the price is below the EMA, it's a good sell
    2. RSI - this is a momentum indicator, in concert with MACD and Williams %R it can be used to determine if the price is going up or down. Our range will as follows:
        - 0 to 30 for oversold
        - 30 to 70 for neutral
        - 70 to 100 for overbought
        In the `ta` library, we can use the `rsi` function to get the RSI indicator. Example: `df.ta.rsi()`
    3. Stochastic Oscillator - this is a momentum indicator
     Triggers for Stochastic Oscillator:
        - If the stochastic oscillator is below 20, it's a good buy
        - If the stochastic oscillator is above 80, it's a good sell
    4. Bollinger Bands - this is a volatility indicator, `ta` has a function called `bbands` that we can use to get the bollinger bands indicator. Example: `df.ta.bbands()`
    Triggers for Bollinger Bands:
        - If the price is above the upper band, it's a good sell
        - If the price is below the lower band, it's a good buy
    5. MACD - this is the moving average convergence divergence indicator. This is a trend indicator. We can use it to determine if the price is going up or down. If the MACD line is above the signal line, it's a good buy. If the MACD line is below the signal line, it's a good sell. Example: `df.ta.macd()`
    Triggers for MACD:
        - If the MACD line is above the signal line, it's a good buy
        - If the MACD line is below the signal line, it's a good sell
    6. Williams %R - this is the inverse of the stochastic oscillator. We can use it to determine if the price is overbought or oversold. If the price is overbought, it's a good sell. If the price is oversold, it's a good buy. Our range will be -20 to 0 for overbought, and -100 to -80 for oversold. To use it `ta` has a function called `willr`. Example: `df.ta.willr()`
    Triggers for Williams %R:
        - If the price is overbought, it's a good sell
        - If the price is oversold, it's a good buy
    7. VWAP - this is the volume weighted average price indicator. It is useful for determining if the price is going up or down. If the price is above the VWAP, it's a good buy. If the price is below the VWAP, it's a good sell. Example: `df.ta.vwap()` What does it stand for? Volume Weighted Average Price. So if we have a coin with a price of $1, and a volume of 100, and a coin with a price of $2, and a volume of 50, then the VWAP would be $1.33. The VWAP is a good indicator of the price of a coin, because it takes into account the volume of the coin. If the volume is high, then the price is likely to be high. If the volume is low, then the price is likely to be low.
    Triggers for VWAP:
        - If the price is above the VWAP, it's a good buy
        - If the price is below the VWAP, it's a good sell
    8. Pivot Points - These are points when the price is likely to reverse. There are 3 types of pivot points: standard, fibonacci, and camarilla. We will use the standard pivot points. Example: `df.ta.pivots()` or `df.ta.pivots(method='standard')` This can be used to determine if the price is going up or down. If the price is above the pivot point, it's a good buy. If the price is below the pivot point, it's a good sell.
    Triggers for Pivot Points:
        - If the price is above the pivot point, it's a good buy
        - If the price is below the pivot point, it's a good sell
    9. Fibonacci Retracement - this is the fibonacci retracement indicator. It's useful for determining support and resistance levels. In the `pandas_ta` library it is named `fib`. In the `ta` library it is named `fibonacci`. We will use `fibonacci`. Example: `df.ta.fibonacci()`
    Triggers for Fibonacci Retracement:
        - If the price is above the 0.618 level, it's a good buy
        - If the price is below the 0.618 level, it's a good sell
    10. Ichimoku Cloud - this is the ichimoku cloud indicator, and is useful for determining if a price movement is real or fake. A cloud in this instance is a region of support or resistance. If the price is going up, but the cloud is below the price, then it's a fake price movement. If the price is going up, and the cloud is above the price, then it's a real price movement.
    Triggers for Ichimoku Cloud:
        - If the price is going up, but the cloud is below the price, then it's a fake price movement, and it means we should likely hold or sell
        - If the price is going up, and the cloud is above the price, then it's a real price movement, and it means we should likely buy
    11. Volume - this is the volume indicator, and is useful for determining if a price movement is real or fake. If the price is going up, but the volume is going down, then it's a fake price movement. If the price is going up, and the volume is going up, then it's a real price movement.
    Triggers for Volume:
        - If the price is going up, but the volume is going down, then it's a fake price movement, and it means we should likely hold or sell
        - If the price is going up, and the volume is going up, then it's a real price movement, and it means we should likely buy
    12. OBV - this is the on balance volume indicator, and is useful for determining if a price movement is real or fake. If the price is going up, but the OBV is going down, then it's a fake price movement. If the price is going up, and the OBV is going up, then it's a real price movement.
    Triggers for OBV:
        - If the price is going up, but the OBV is going down, then it's a fake price movement, and it means we should likely hold or sell
        - If the price is going up, and the OBV is going up, then it's a real price movement, and it means we should likely buy
    13. ATR - this is the average true range indicator, and is useful for determining the volatility of a coin. If the ATR is high, then the coin is volatile. If the ATR is low, then the coin is not volatile.
    We can use this to determine if we should buy or sell. If the ATR is high, then we should buy. If the ATR is low, then we should sell.
    Triggers for ATR:
        - If the ATR is high, then we should buy
        - If the ATR is low, then we should sell