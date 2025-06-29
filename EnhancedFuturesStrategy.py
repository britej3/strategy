# pragma pylint: disable=missing-docstring, invalid-name, too-few-public-methods, unnecessary-lambda
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np
import pandas as pd
from functools import reduce
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

# Freqtrade imports for strategy and hyperopt parameters
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, IntParameter, IStrategy)
from freqtrade.enums import CandleType

# Import TA-Lib for technical indicators
import talib.abstract as ta 
# Removed: from talib import MA_Type - now using integer constant directly


# SSL Channels - Helper function, moved outside the class for cleaner integration
def SSLChannels(dataframe: pd.DataFrame, length: int = 7) -> Tuple[pd.Series, pd.Series]:
    """
    Calculates SSL Channels.
    Adapted from your original BinClucMadV1.py logic.
    """
    df = dataframe.copy()
    # Ensure 'high', 'low', 'close' are numeric
    for col in ['high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # ATR requires 'high', 'low', 'close'
    if not all(col in df.columns for col in ['high', 'low', 'close']):
        # If columns are missing, return series of NaNs with the correct index
        return pd.Series(np.nan, index=df.index), pd.Series(np.nan, index=df.index)

    df["ATR"] = ta.ATR(df["high"], df["low"], df["close"], timeperiod=14)
    df["smaHigh"] = df["high"].rolling(length).mean() + df["ATR"]
    df["smaLow"] = df["low"].rolling(length).mean() - df["ATR"]
    df["hlv"] = np.where(df["close"] > df["smaHigh"], 1, np.where(df["close"] < df["smaLow"], -1, np.NAN))
    df["hlv"] = df["hlv"].ffill()
    df["sslDown"] = np.where(df["hlv"] < 0, df["smaHigh"], df["smaLow"])
    df["sslUp"] = np.where(df["hlv"] < 0, df["smaLow"], df["smaHigh"])
    return df["sslDown"], df["sslUp"]


class EnhancedFuturesStrategy(IStrategy):
    """
    Enhanced Freqtrade strategy for Crypto Futures trading (2025 Standards).
    This strategy supports both Long and Short positions with funding rate optimization
    and dynamic leverage management.
    """
    # Strategy interface version - Latest as of 2025
    INTERFACE_VERSION = 3

    # Optimal timeframe and informative timeframe
    timeframe = '5m'
    informative_timeframe = '1h'
    funding_rate_timeframe = '8h'  # Binance's standard funding interval

    # Enable both long and short trading
    can_short = True

    # Minimal ROI - Will be optimized by hyperopt
    minimal_roi = {
        "0": 0.038,
        "10": 0.028,
        "40": 0.015,
        "180": 0.018,
    }

    # Stoploss (negative percentage)
    stoploss = -0.99  # Effectively disabled when custom_stoploss is used

    # Modern exit signal handling (2025 standards)
    use_exit_signal = True
    exit_profit_only = False
    exit_profit_offset = 0.001
    ignore_roi_if_entry_signal = False # Important for Futures, allows custom exit to override ROI table

    # Custom stoploss and exit control
    use_custom_stoploss = True
    use_custom_exit = True # Enable custom exit to manage both long and short exits

    # Process only new candles for efficiency
    process_only_new_candles = True

    # Startup candles for indicator stability
    startup_candle_count: int = 220  # Increased for better indicator accuracy (SMA 200 + 20 shift)

    ############################################################################
    # HYPEROPT PARAMETERS - All organized by optimization spaces
    ############################################################################

    # === LEVERAGE OPTIMIZATION SPACE ===
    dynamic_leverage_enable = BooleanParameter(default=True, space="leverage", optimize=True)
    volatility_threshold_high = DecimalParameter(0.03, 0.1, default=0.05, space="leverage", decimals=3, optimize=True)
    volatility_threshold_low = DecimalParameter(0.01, 0.04, default=0.02, space="leverage", decimals=3, optimize=True)
    leverage_high_vol = IntParameter(2, 8, default=5, space="leverage", optimize=True)
    leverage_low_vol = IntParameter(7, 15, default=10, space="leverage", optimize=True)
    default_leverage = IntParameter(5, 12, default=7, space="leverage", optimize=True)

    # === FUNDING RATE OPTIMIZATION SPACE (Part of Protection or its own space) ===
    # Moved to 'protection' space as it's a filter/protection mechanism
    use_funding_rate_filter = BooleanParameter(default=True, space="protection", optimize=True)
    funding_rate_long_threshold = DecimalParameter(-0.001, 0.002, default=0.0005, space="protection", decimals=5, optimize=True)
    funding_rate_short_threshold = DecimalParameter(-0.002, 0.001, default=-0.0005, space="protection", decimals=5, optimize=True)

    # === LONG ENTRY OPTIMIZATION SPACE (Mapped to 'buy' for hyperopt compatibility) ===
    # Curve-fitting guards: Narrowed ranges to prevent overfitting
    long_condition_0_enable = BooleanParameter(default=True, space="buy", optimize=True) 
    long_condition_1_enable = BooleanParameter(default=True, space="buy", optimize=True)
    long_condition_2_enable = BooleanParameter(default=True, space="buy", optimize=True)
    long_condition_3_enable = BooleanParameter(default=True, space="buy", optimize=True)
    long_condition_4_enable = BooleanParameter(default=False, space="buy", optimize=True)

    # Long entry thresholds with constrained ranges to avoid curve-fitting
    long_dip_threshold_0 = DecimalParameter(0.01, 0.05, default=0.015, space="buy", decimals=3, optimize=True)
    long_dip_threshold_1 = DecimalParameter(0.1, 0.15, default=0.12, space="buy", decimals=2, optimize=True)
    long_dip_threshold_2 = DecimalParameter(0.2, 0.3, default=0.28, space="buy", decimals=2, optimize=True)
    long_dip_threshold_3 = DecimalParameter(0.3, 0.4, default=0.36, space="buy", decimals=2, optimize=True)
    long_bb40_bbdelta_close = DecimalParameter(0.02, 0.035, default=0.031, space="buy", optimize=True)
    long_bb40_closedelta_close = DecimalParameter(0.015, 0.025, default=0.021, space="buy", optimize=True)
    long_bb40_tail_bbdelta = DecimalParameter(0.25, 0.35, default=0.264, space="buy", optimize=True)
    long_bb20_close_bblowerband = DecimalParameter(0.9, 1.05, default=0.992, space="buy", optimize=True)
    long_bb20_volume = IntParameter(20, 30, default=29, space="buy", optimize=True)
    long_rsi_diff = DecimalParameter(40.0, 55.0, default=50.48, space="buy", decimals=2, optimize=True)
    long_min_inc = DecimalParameter(0.008, 0.02, default=0.01, space="buy", decimals=2, optimize=True)
    long_rsi_1h = DecimalParameter(50.0, 65.0, default=67.0, space="buy", decimals=2, optimize=True)
    long_rsi = DecimalParameter(32.0, 38.0, default=38.5, space="buy", decimals=2, optimize=True)
    long_mfi = DecimalParameter(40.0, 50.0, default=36.0, space="buy", decimals=2, optimize=True)

    # === SHORT ENTRY OPTIMIZATION SPACE (Mapped to 'buy' for hyperopt compatibility) ===
    # For short entries, Freqtrade hyperopt still uses 'buy' space for these parameters when can_short = True
    # Curve-fitting guards: Narrowed ranges to prevent overfitting
    short_condition_0_enable = BooleanParameter(default=True, space="buy", optimize=True) 
    short_condition_1_enable = BooleanParameter(default=True, space="buy", optimize=True)
    short_condition_2_enable = BooleanParameter(default=True, space="buy", optimize=True)
    short_condition_3_enable = BooleanParameter(default=True, space="buy", optimize=True) 
    short_condition_4_enable = BooleanParameter(default=True, space="buy", optimize=True) 

    # Short entry thresholds (inverse logic of longs) with constrained ranges to avoid curve-fitting
    short_pump_threshold_1 = DecimalParameter(0.1, 0.15, default=0.12, space="buy", decimals=2, optimize=True)
    short_pump_threshold_2 = DecimalParameter(0.2, 0.3, default=0.28, space="buy", decimals=2, optimize=True)
    short_bb40_bbdelta_close = DecimalParameter(0.02, 0.035, default=0.031, space="buy", optimize=True)
    short_bb20_close_bbupperband = DecimalParameter(0.95, 1.1, default=1.008, space="buy", optimize=True)
    short_rsi_1h = DecimalParameter(35.0, 50.0, default=33.0, space="buy", decimals=2, optimize=True)
    short_rsi = DecimalParameter(65.0, 75.0, default=70.0, space="buy", decimals=2, optimize=True)
    short_mfi = DecimalParameter(70.0, 85.0, default=80.0, space="buy", decimals=2, optimize=True)
    short_ema_open_mult = DecimalParameter(0.015, 0.03, default=0.02, space="buy", decimals=3, optimize=True) 
    short_volume_spike = DecimalParameter(1.8, 2.5, default=2.0, space="buy", decimals=1, optimize=True) 

    # === EXIT OPTIMIZATION SPACE (Mapped to 'sell' for hyperopt compatibility) ===
    # For long positions
    exit_long_rsi_main = DecimalParameter(70.0, 90.0, default=80, space="sell", optimize=True) 
    
    # For short positions
    exit_short_rsi_main = DecimalParameter(10.0, 30.0, default=20, space="sell", optimize=True) 
    
    # Custom ROI exit parameters (apply to both long/short symmetrically)
    custom_roi_profit_1 = DecimalParameter(0.005, 0.03, default=0.01, space="sell", decimals=3, optimize=True) 
    custom_roi_rsi_1 = DecimalParameter(40.0, 56.0, default=50, space="sell", decimals=2, optimize=True)
    custom_roi_profit_2 = DecimalParameter(0.01, 0.1, default=0.04, space="sell", decimals=2, optimize=True)
    custom_roi_rsi_2 = DecimalParameter(42.0, 56.0, default=50, space="sell", decimals=2, optimize=True)
    custom_roi_profit_3 = DecimalParameter(0.05, 0.20, default=0.08, space="sell", decimals=2, optimize=True)
    custom_roi_rsi_3 = DecimalParameter(44.0, 58.0, default=56, space="sell", decimals=2, optimize=True)
    custom_roi_profit_4 = DecimalParameter(0.1, 0.4, default=0.14, space="sell", decimals=2, optimize=True)
    custom_roi_rsi_4 = DecimalParameter(44.0, 60.0, default=58, space="sell", decimals=2, optimize=True)
    # New custom profit exit for both (if in positive profit and SMA is decreasing for long, increasing for short)
    custom_roi_profit_aggressive = DecimalParameter(0.01, 0.1, default=0.04, space="sell", decimals=2, optimize=True)


    # === PROTECTION OPTIMIZATION SPACE ===
    custom_stoploss_long_1 = DecimalParameter(-0.15, -0.01, default=-0.05, space="protection", decimals=2, optimize=True)
    custom_stoploss_short_1 = DecimalParameter(-0.15, -0.01, default=-0.05, space="protection", decimals=2, optimize=True)


    ############################################################################
    # CORE STRATEGY METHODS
    ############################################################################

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Calculates all necessary indicators for the strategy.
        Enhanced with better error handling and data validation.
        """
        # Ensure all OHLCV columns are numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in dataframe.columns:
                dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')

        # === INFORMATIVE 1H INDICATORS ===
        try:
            informative_1h = self.dp.get_pair_dataframe(
                pair=metadata['pair'], 
                timeframe=self.informative_timeframe
            )
            
            if not informative_1h.empty and len(informative_1h) >= self.startup_candle_count:
                # Ensure numeric columns
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in informative_1h.columns:
                        informative_1h[col] = pd.to_numeric(informative_1h[col], errors='coerce')
                
                # Calculate 1h indicators
                informative_1h["ema_50"] = ta.EMA(informative_1h["close"], timeperiod=50)
                informative_1h["ema_100"] = ta.EMA(informative_1h["close"], timeperiod=100)
                informative_1h["ema_200"] = ta.EMA(informative_1h["close"], timeperiod=200)
                informative_1h["sma_200"] = ta.SMA(informative_1h["close"], timeperiod=200)
                
                # SMA trend detection
                if len(informative_1h) >= 20: 
                    informative_1h["sma_200_dec"] = informative_1h["sma_200"] < informative_1h["sma_200"].shift(20)
                    informative_1h["sma_200_inc"] = informative_1h["sma_200"] > informative_1h["sma_200"].shift(20)
                else:
                    informative_1h["sma_200_dec"] = False
                    informative_1h["sma_200_inc"] = False
                     
                informative_1h["rsi"] = ta.RSI(informative_1h["close"], timeperiod=14)
                
                # SSL Channels
                ssl_down_1h, ssl_up_1h = SSLChannels(informative_1h, 20)
                informative_1h["ssl_down"] = ssl_down_1h
                informative_1h["ssl_up"] = ssl_up_1h
                informative_1h["ssl_dir"] = np.where(ssl_up_1h > ssl_down_1h, "up", "down")
                informative_1h["ATR"] = ta.ATR(informative_1h["high"], informative_1h["low"], informative_1h["close"], timeperiod=14)

                # Merge informative data manually using pandas
                informative_1h = informative_1h.add_suffix('_1h')
                informative_1h['date'] = informative_1h['date_1h']
                dataframe = dataframe.merge(informative_1h, on='date', how='left').fillna(method='ffill')
                
        except Exception as e:
            print(f"WARNING: Failed to fetch informative 1h data for {metadata['pair']}: {e}") # Using print as fallback
            # Ensure informative columns exist even if data fetching fails
            for col_suffix in ['_1h']:
                for base_col in ['ema_50', 'ema_100', 'ema_200', 'sma_200', 'sma_200_dec', 'sma_200_inc', 'rsi', 'ssl_down', 'ssl_up', 'ssl_dir', 'ATR']:
                    if (base_col + col_suffix) not in dataframe.columns:
                        dataframe[base_col + col_suffix] = np.nan


        # === FUNDING RATE DATA ===
        try:
            funding_rate_df = self.dp.get_pair_dataframe(
                pair=metadata['pair'],
                timeframe=self.funding_rate_timeframe,
                candle_type=CandleType.FUNDING_RATE
            )

            if not funding_rate_df.empty:
                funding_rate_df['rate'] = pd.to_numeric(funding_rate_df['rate'], errors='coerce').ffill()
                
                # Merge funding rate data manually using pandas
                funding_rate_df = funding_rate_df[['rate']].add_suffix('_funding')
                funding_rate_df['date'] = funding_rate_df.index
                dataframe = dataframe.merge(funding_rate_df, on='date', how='left').fillna(method='ffill')
                dataframe.rename(columns={'rate_funding': 'funding_rate'}, inplace=True)
                dataframe['funding_rate'] = pd.to_numeric(dataframe['funding_rate'], errors='coerce').fillna(0.0)
            else:
                dataframe['funding_rate'] = 0.0 # Default to 0 if no funding rate data
                
        except Exception as e:
            print(f"WARNING: Failed to fetch funding rate data for {metadata['pair']}: {e}") # Using print as fallback
            dataframe['funding_rate'] = 0.0 # Ensure column exists if fetch fails


        # === MAIN TIMEFRAME INDICATORS ===
        
        # Bollinger Bands (40 period)
        if len(dataframe) >= 40:
            # Use 0 for SMA type instead of MA_Type.SMA
            bb_40 = ta.BBANDS(dataframe["close"], timeperiod=40, nbdevup=2.0, nbdevdn=2.0, matype=0) 
            dataframe["lower"] = bb_40[0]
            dataframe["mid"] = bb_40[1]
            dataframe["upper_40"] = bb_40[2]
            dataframe["bbdelta"] = (dataframe["mid"] - dataframe["lower"]).abs()
        else:
            for col in ["lower", "mid", "upper_40", "bbdelta"]:
                dataframe[col] = np.nan

        # Price deltas and candle analysis
        dataframe["closedelta"] = (dataframe["close"] - dataframe["close"].shift(1)).abs()
        dataframe["tail"] = (dataframe["close"] - dataframe["low"]).abs()
        dataframe["wick"] = (dataframe["high"] - dataframe["close"]).abs() # Upper wick for short analysis

        # Bollinger Bands (20 period)
        if len(dataframe) >= 20:
            # Use 0 for SMA type instead of MA_Type.SMA
            bollinger = ta.BBANDS(dataframe["close"], timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0) 
            dataframe["bb_lowerband"] = bollinger[0]
            dataframe["bb_middleband"] = bollinger[1]
            dataframe["bb_upperband"] = bollinger[2]
        else:
            for col in ["bb_lowerband", "bb_middleband", "bb_upperband"]:
                dataframe[col] = np.nan

        # Volume analysis
        if len(dataframe) >= 30:
            dataframe["volume_mean_slow"] = dataframe["volume"].rolling(window=30).mean()
        else:
            dataframe["volume_mean_slow"] = np.nan

        # Moving averages
        dataframe["ema_12"] = ta.EMA(dataframe["close"], timeperiod=12)
        dataframe["ema_26"] = ta.EMA(dataframe["close"], timeperiod=26)
        dataframe["ema_50"] = ta.EMA(dataframe["close"], timeperiod=50)
        dataframe["ema_200"] = ta.EMA(dataframe["close"], timeperiod=200)
        dataframe["sma_5"] = ta.SMA(dataframe["close"], timeperiod=5)
        dataframe["sma_200"] = ta.SMA(dataframe["close"], timeperiod=200)
        
        # SMA trend for main timeframe
        if len(dataframe) >= 220:
            dataframe["sma_200_dec"] = dataframe["sma_200"] < dataframe["sma_200"].shift(20)
            dataframe["sma_200_inc"] = dataframe["sma_200"] > dataframe["sma_200"].shift(20)
        else:
            dataframe["sma_200_dec"] = False
            dataframe["sma_200_inc"] = False


        # Oscillators
        dataframe["rsi"] = ta.RSI(dataframe["close"], timeperiod=14)
        dataframe["mfi"] = ta.MFI(dataframe["high"], dataframe["low"], dataframe["close"], dataframe["volume"], timeperiod=14)
        
        # Volatility calculation for dynamic leverage (using ATR from main timeframe if 1h ATR not available)
        if 'ATR_1h' not in dataframe.columns or dataframe['ATR_1h'].isnull().all():
            if len(dataframe) >= 14: # ATR requires 14 periods
                dataframe["ATR"] = ta.ATR(dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=14)
            else:
                dataframe["ATR"] = np.nan
        else:
            dataframe["ATR"] = dataframe["ATR_1h"] # Use 1h ATR if available and merged

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Populate entry signals for both long and short positions.
        This method replaces the deprecated populate_buy_trend.
        """
        long_conditions = []
        short_conditions = []

        # Validate that all necessary columns for conditions are present and not all NaN
        required_cols = [
            "close", "ema_200", "ema_50", "bb_lowerband", "bb_upperband", 
            "volume_mean_slow", "volume", "open", "ema_26", "ema_12", 
            "rsi", "lower", "bbdelta", "closedelta", "tail", "wick", "sma_5", "mfi",
            "ema_200_1h", "rsi_1h", "ema_50_1h", "ema_100_1h", 
            "sma_200_1h", "ssl_up_1h", "ssl_down_1h", "ATR_1h", "funding_rate",
            "sma_200_dec", "sma_200_inc" # Ensure these are present for latest conditions
        ]
        
        for col in required_cols:
            if col not in dataframe.columns or dataframe[col].isnull().all():
                print(f"WARNING: Missing or all-NaN column '{col}' for entry trend calculation for {metadata['pair']}. Skipping entry signals.")
                dataframe.loc[:, 'enter_long'] = 0
                dataframe.loc[:, 'enter_short'] = 0
                return dataframe

        # === LONG ENTRY CONDITIONS ===
        # Long Condition 0: Oversold bounce near EMA support
        if self.long_condition_0_enable.value and len(dataframe) >= 12:
            cond_0 = (
                (dataframe["close"] > dataframe["ema_200_1h"])
                & (dataframe["ema_50"] > dataframe["ema_200"])
                & (dataframe["ema_50_1h"] > dataframe["ema_200_1h"])
                & (((dataframe["open"].rolling(2).max() - dataframe["close"]) / dataframe["close"]) < self.long_dip_threshold_1.value)
                & (((dataframe["open"].rolling(12).max() - dataframe["close"]) / dataframe["close"]) < self.long_dip_threshold_2.value)
                & (dataframe["bbdelta"] > dataframe["close"] * self.long_bb40_bbdelta_close.value)
                & (dataframe["closedelta"] > dataframe["close"] * self.long_bb40_closedelta_close.value)
                & (dataframe["tail"] < dataframe["bbdelta"] * self.long_bb40_tail_bbdelta.value)
                & (dataframe["close"] < dataframe["lower"].shift(1))
                & (dataframe["close"] <= dataframe["close"].shift(1)) # Current candle not significantly higher
                & (dataframe["volume"] > 0)
            )
            long_conditions.append(cond_0)

        # Long Condition 1: BB lower band bounce
        if self.long_condition_1_enable.value and len(dataframe) >= 12:
            cond_1 = (
                (dataframe["close"] > dataframe["ema_200"])
                & (dataframe["close"] > dataframe["ema_200_1h"])
                & (dataframe["ema_50_1h"] > dataframe["ema_100_1h"])
                & (dataframe["ema_50_1h"] > dataframe["ema_200_1h"])
                & (((dataframe["open"].rolling(2).max() - dataframe["close"]) / dataframe["close"]) < self.long_dip_threshold_1.value)
                & (((dataframe["open"].rolling(12).max() - dataframe["close"]) / dataframe["close"]) < self.long_dip_threshold_2.value)
                & (dataframe["close"] < dataframe["ema_50"])
                & (dataframe["close"] < self.long_bb20_close_bblowerband.value * dataframe["bb_lowerband"])
                & (dataframe["volume"] < (dataframe["volume_mean_slow"].shift(1) * self.long_bb20_volume.value))
                & (dataframe["volume"] > 0)
            )
            long_conditions.append(cond_1)

        # Long Condition 2: SSL channel uptrend with RSI divergence
        if self.long_condition_2_enable.value and len(dataframe) >= 144:
            cond_2 = (
                (dataframe["close"] < dataframe["sma_5"])
                & (dataframe["ssl_up_1h"] > dataframe["ssl_down_1h"])
                & (dataframe["ema_50"] > dataframe["ema_200"])
                & (dataframe["ema_50_1h"] > dataframe["ema_200_1h"])
                & (((dataframe["open"].rolling(2).max() - dataframe["close"]) / dataframe["close"]) < self.long_dip_threshold_1.value)
                & (((dataframe["open"].rolling(12).max() - dataframe["close"]) / dataframe["close"]) < self.long_dip_threshold_2.value)
                & (((dataframe["open"].rolling(144).max() - dataframe["close"]) / dataframe["close"]) < self.long_dip_threshold_3.value)
                & (dataframe["rsi"] < dataframe["rsi_1h"] - self.long_rsi_diff.value)
                & (dataframe["volume"] > 0)
            )
            long_conditions.append(cond_2)

        # Long Condition 3: Strong uptrend with pullback and MFI confirmation
        if self.long_condition_3_enable.value and len(dataframe) >= 24:
            rolling_open_24_min = dataframe["open"].rolling(24).min()
            cond_3 = (
                (dataframe["sma_200"] > dataframe["sma_200"].shift(20)) # SMA 200 increasing on 5m
                & (dataframe["sma_200_1h"] > dataframe["sma_200_1h"].shift(16)) # SMA 200 increasing on 1h
                & (((dataframe["open"].rolling(2).max() - dataframe["close"]) / dataframe["close"]) < self.long_dip_threshold_1.value)
                & (((dataframe["open"].rolling(12).max() - dataframe["close"]) / dataframe["close"]) < self.long_dip_threshold_2.value)
                & (((dataframe["open"].rolling(144).max() - dataframe["close"]) / dataframe["close"]) < self.long_dip_threshold_3.value)
                & (((rolling_open_24_min - dataframe["close"]) / dataframe["close"]) > self.long_min_inc.value)
                & (dataframe["rsi_1h"] > self.long_rsi_1h.value)
                & (dataframe["rsi"] < self.long_rsi.value) # Pullback on 5m RSI
                & (dataframe["mfi"] < self.long_mfi.value) # MFI oversold
                & (dataframe["volume"] > 0)
            )
            long_conditions.append(cond_3)

        # Long Condition 4: EMA cross and price below BB lower band (aggressive)
        if self.long_condition_4_enable.value and len(dataframe) >= 144:
            cond_4 = (
                (dataframe["close"] > dataframe["ema_100_1h"]) # Price above 1h EMA100
                & (dataframe["ema_50_1h"] > dataframe["ema_100_1h"]) # EMAs supporting uptrend on 1h
                & (((dataframe["open"].rolling(2).max() - dataframe["close"]) / dataframe["close"]) < self.long_dip_threshold_1.value)
                & (((dataframe["open"].rolling(12).max() - dataframe["close"]) / dataframe["close"]) < self.long_dip_threshold_2.value)
                & (((dataframe["open"].rolling(144).max() - dataframe["close"]) / dataframe["close"]) < self.long_dip_threshold_3.value)
                & (dataframe["volume"].rolling(4).mean() * self.long_bb20_volume.value > dataframe["volume"]) # Volume condition
                & (dataframe["ema_26"] > dataframe["ema_12"]) # EMA cross
                & ((dataframe["ema_26"] - dataframe["ema_12"]) > (dataframe["open"] * self.long_bb40_bbdelta_close.value)) # MACD strength
                & ((dataframe["ema_26"].shift(1) - dataframe["ema_12"].shift(1)) > (dataframe["open"] / 100))
                & (dataframe["close"] < (dataframe["bb_lowerband"])) # Price below BB lower band
                & (dataframe["volume"] > 0)
            )
            long_conditions.append(cond_4)


        # === SHORT ENTRY CONDITIONS (Inverse Logic of Longs) ===
        
        # Short Condition 0: Overbought rejection near EMA resistance
        if self.short_condition_0_enable.value and len(dataframe) >= 12:
            cond_0 = (
                (dataframe["close"] < dataframe["ema_200_1h"]) # Price below 1h EMA200
                & (dataframe["ema_50"] < dataframe["ema_200"]) # 5m EMA50 below EMA200
                & (dataframe["ema_50_1h"] < dataframe["ema_200_1h"]) # 1h EMA50 below EMA200
                & (((dataframe["close"] - dataframe["open"].rolling(2).min()) / dataframe["close"]) < self.short_pump_threshold_1.value) # Price pump check
                & (((dataframe["close"] - dataframe["open"].rolling(12).min()) / dataframe["close"]) < self.short_pump_threshold_2.value) # Price pump check
                & (dataframe["bbdelta"] > dataframe["close"] * self.short_bb40_bbdelta_close.value) # BB width
                & (dataframe["wick"] < dataframe["bbdelta"] * self.long_bb40_tail_bbdelta.value) # Small upper wick (rejection)
                & (dataframe["close"] > dataframe["upper_40"].shift(1)) # Price above previous BB upper
                & (dataframe["close"] >= dataframe["close"].shift(1)) # Current candle not significantly lower
                & (dataframe["volume"] > 0)
            )
            short_conditions.append(cond_0)

        # Short Condition 1: BB upper band rejection
        if self.short_condition_1_enable.value and len(dataframe) >= 12:
            cond_1 = (
                (dataframe["close"] < dataframe["ema_200"]) # Price below 5m EMA200
                & (dataframe["close"] < dataframe["ema_200_1h"]) # Price below 1h EMA200
                & (dataframe["ema_50_1h"] < dataframe["ema_100_1h"]) # 1h EMAs supporting downtrend
                & (dataframe["ema_50_1h"] < dataframe["ema_200_1h"])
                & (((dataframe["close"] - dataframe["open"].rolling(2).min()) / dataframe["close"]) < self.short_pump_threshold_1.value)
                & (((dataframe["close"] - dataframe["open"].rolling(12).min()) / dataframe["close"]) < self.short_pump_threshold_2.value)
                & (dataframe["close"] > dataframe["ema_50"]) # Price above 5m EMA50
                & (dataframe["close"] > self.short_bb20_close_bbupperband.value * dataframe["bb_upperband"]) # Price above BB upper band
                & (dataframe["rsi"] > self.short_rsi.value) # Overbought RSI
                & (dataframe["rsi_1h"] > self.short_rsi_1h.value) # Overbought 1h RSI
                & (dataframe["volume"] > 0)
            )
            short_conditions.append(cond_1)

        # Short Condition 2: High RSI with volume spike and MFI confirmation
        if self.short_condition_2_enable.value:
            cond_2 = (
                (dataframe["rsi"] > self.short_rsi.value)
                & (dataframe["rsi_1h"] > self.short_rsi_1h.value)
                & (dataframe["mfi"] > self.short_mfi.value)
                & (dataframe["close"] > dataframe["bb_upperband"])
                & (dataframe["volume"] > dataframe["volume_mean_slow"] * self.short_volume_spike.value) # Volume spike
                & (dataframe["volume"] > 0)
            )
            short_conditions.append(cond_2)

        # Short Condition 3: Strong downtrend with bounce towards MAs (inverse of Long 3)
        if self.short_condition_3_enable.value and len(dataframe) >= 24:
            rolling_open_24_max = dataframe["open"].rolling(24).max()
            cond_3 = (
                (dataframe["sma_200"] < dataframe["sma_200"].shift(20)) # SMA 200 decreasing on 5m
                & (dataframe["sma_200_1h"] < dataframe["sma_200_1h"].shift(16)) # SMA 200 decreasing on 1h
                & (((dataframe["close"] - dataframe["open"].rolling(2).min()) / dataframe["close"]) < self.short_pump_threshold_1.value)
                & (((dataframe["close"] - dataframe["open"].rolling(12).min()) / dataframe["close"]) < self.short_pump_threshold_2.value)
                & (((dataframe["close"] - rolling_open_24_max) / dataframe["close"]) < -self.long_min_inc.value) # Price increased from 24-candle high (bounce)
                & (dataframe["rsi_1h"] < self.short_rsi_1h.value) # 1h RSI oversold
                & (dataframe["rsi"] > self.short_rsi.value) # 5m RSI overbought (bounce)
                & (dataframe["mfi"] > self.short_mfi.value) # MFI overbought
                & (dataframe["volume"] > 0)
            )
            short_conditions.append(cond_3)

        # Short Condition 4: EMA cross and price above BB upper band (aggressive, inverse of Long 4)
        if self.short_condition_4_enable.value and len(dataframe) >= 144:
            cond_4 = (
                (dataframe["close"] < dataframe["ema_100_1h"]) # Price below 1h EMA100
                & (dataframe["ema_50_1h"] < dataframe["ema_100_1h"]) # EMAs supporting downtrend on 1h
                & (((dataframe["close"] - dataframe["open"].rolling(2).min()) / dataframe["close"]) < self.short_pump_threshold_1.value)
                & (((dataframe["close"] - dataframe["open"].rolling(12).min()) / dataframe["close"]) < self.short_pump_threshold_2.value)
                & (dataframe["volume"].rolling(4).mean() * self.long_bb20_volume.value > dataframe["volume"]) # Volume condition (inverse interpretation)
                & (dataframe["ema_26"] < dataframe["ema_12"]) # EMA cross (bearish)
                & ((dataframe["ema_12"] - dataframe["ema_26"]) > (dataframe["open"] * self.short_ema_open_mult.value)) # MACD strength
                & ((dataframe["ema_12"].shift(1) - dataframe["ema_26"].shift(1)) > (dataframe["open"] / 100))
                & (dataframe["close"] > (dataframe["bb_upperband"])) # Price above BB upper band
                & (dataframe["volume"] > 0)
            )
            short_conditions.append(cond_4)


        # Apply funding rate filters if enabled
        if self.use_funding_rate_filter.value:
            if 'funding_rate' in dataframe.columns and not dataframe['funding_rate'].isnull().all():
                # For long entry, funding rate should be below or equal to threshold (avoid high positive funding)
                long_funding_cond = (dataframe['funding_rate'] <= self.funding_rate_long_threshold.value)
                long_conditions = [cond & long_funding_cond for cond in long_conditions]
                
                # For short entry, funding rate should be above or equal to threshold (avoid high negative funding)
                short_funding_cond = (dataframe['funding_rate'] >= self.funding_rate_short_threshold.value)
                short_conditions = [cond & short_funding_cond for cond in short_conditions]
            else:
                print(f"WARNING: Funding rate data not available for entry filter for {metadata['pair']}. Filter skipped.")

        # Combine all conditions
        if long_conditions:
            dataframe.loc[reduce(lambda x, y: x | y, long_conditions), 'enter_long'] = 1
        else:
            dataframe.loc[:, 'enter_long'] = 0

        if short_conditions:
            dataframe.loc[reduce(lambda x, y: x | y, short_conditions), 'enter_short'] = 1
        else:
            dataframe.loc[:, 'enter_short'] = 0

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Populate exit signals for both long and short positions.
        This method replaces the deprecated populate_sell_trend.
        """
        long_exit_conditions = []
        short_exit_conditions = []

        # Validate that all necessary columns for conditions are present and not all NaN
        required_cols = [
            "close", "bb_middleband", "volume", "bb_upperband", "bb_lowerband", "rsi",
            "sma_200_dec", "sma_200_inc", "rsi_1h" 
        ]
        
        for col in required_cols:
            if col not in dataframe.columns or dataframe[col].isnull().all():
                print(f"WARNING: Missing or all-NaN column '{col}' for exit trend calculation for {metadata['pair']}. Skipping exit signals.")
                dataframe.loc[:, 'exit_long'] = 0
                dataframe.loc[:, 'exit_short'] = 0
                return dataframe

        # === LONG EXIT CONDITIONS ===
        # Long Exit 0: Price crosses below BB middle band (aggressive)
        if len(dataframe) >= 1:
            cond_0_long = (
                (dataframe["close"] < dataframe["bb_middleband"])
                & (dataframe["volume"] > 0)
            )
            long_exit_conditions.append(cond_0_long)

        # Long Exit 1: Price crosses above high RSI (overbought)
        if len(dataframe) >= 2:
            cond_1_long = (
                (dataframe["rsi"].shift(1) < self.exit_long_rsi_main.value)
                & (dataframe["rsi"] >= self.exit_long_rsi_main.value)
                & (dataframe["volume"] > 0)
            )
            long_exit_conditions.append(cond_1_long)
        
        # --- Custom ROI/RSI exits for LONG positions ---
        # Note: These are typically handled in `custom_exit` based on trade profit,
        # but included here for potential combined logic or backtesting visibility.
        cond_roi_long_1 = (dataframe["close"].pct_change(fill_method=None) > self.custom_roi_profit_1.value) & (dataframe["rsi"] < self.custom_roi_rsi_1.value)
        cond_roi_long_2 = (dataframe["close"].pct_change(fill_method=None) > self.custom_roi_profit_2.value) & (dataframe["rsi"] < self.custom_roi_rsi_2.value)
        cond_roi_long_3 = (dataframe["close"].pct_change(fill_method=None) > self.custom_roi_profit_3.value) & (dataframe["rsi"] < self.custom_roi_rsi_3.value)
        cond_roi_long_4 = (dataframe["close"].pct_change(fill_method=None) > self.custom_roi_profit_4.value) & (dataframe["rsi"] < self.custom_roi_rsi_4.value)
        
        # Aggressive exit if profit is positive and SMA200 is decreasing
        cond_aggressive_long = (dataframe["close"].pct_change(fill_method=None) > 0) & (dataframe["close"].pct_change(fill_method=None) < self.custom_roi_profit_aggressive.value) & (dataframe["sma_200_dec"])
        
        long_exit_conditions.extend([cond_roi_long_1, cond_roi_long_2, cond_roi_long_3, cond_roi_long_4, cond_aggressive_long])


        # === SHORT EXIT CONDITIONS (Inverse Logic of Long Exits) ===
        # Short Exit 0: Price crosses above BB middle band (aggressive)
        if len(dataframe) >= 1:
            cond_0_short = (
                (dataframe["close"] > dataframe["bb_middleband"])
                & (dataframe["volume"] > 0)
            )
            short_exit_conditions.append(cond_0_short)
        
        # Short Exit 1: Price crosses below low RSI (oversold)
        if len(dataframe) >= 2:
            cond_1_short = (
                (dataframe["rsi"].shift(1) > self.exit_short_rsi_main.value)
                & (dataframe["rsi"] <= self.exit_short_rsi_main.value)
                & (dataframe["volume"] > 0)
            )
            short_exit_conditions.append(cond_1_short)

        # --- Custom ROI/RSI exits for SHORT positions ---
        # Inverse logic for profit percentages and RSI thresholds
        cond_roi_short_1 = (dataframe["close"].pct_change(fill_method=None) < -self.custom_roi_profit_1.value) & (dataframe["rsi"] > (100 - self.custom_roi_rsi_1.value))
        cond_roi_short_2 = (dataframe["close"].pct_change(fill_method=None) < -self.custom_roi_profit_2.value) & (dataframe["rsi"] > (100 - self.custom_roi_rsi_2.value))
        cond_roi_short_3 = (dataframe["close"].pct_change(fill_method=None) < -self.custom_roi_profit_3.value) & (dataframe["rsi"] > (100 - self.custom_roi_rsi_3.value))
        cond_roi_short_4 = (dataframe["close"].pct_change(fill_method=None) < -self.custom_roi_profit_4.value) & (dataframe["rsi"] > (100 - self.custom_roi_rsi_4.value))

        # Aggressive exit for short if profit is positive (for short, this means price dropped) and SMA200 is increasing
        cond_aggressive_short = (dataframe["close"].pct_change(fill_method=None) < 0) & (dataframe["close"].pct_change(fill_method=None) > -self.custom_roi_profit_aggressive.value) & (dataframe["sma_200_inc"])

        short_exit_conditions.extend([cond_roi_short_1, cond_roi_short_2, cond_roi_short_3, cond_roi_short_4, cond_aggressive_short])


        # Apply combined conditions
        if long_exit_conditions:
            dataframe.loc[reduce(lambda x, y: x | y, long_exit_conditions), 'exit_long'] = 1
        else:
            dataframe.loc[:, 'exit_long'] = 0

        if short_exit_conditions:
            dataframe.loc[reduce(lambda x, y: x | y, short_exit_conditions), 'exit_short'] = 1 
        else:
            dataframe.loc[:, 'exit_short'] = 0

        return dataframe

    def custom_stake(self, pair: str, current_time: datetime, current_rate: float,
                     candle: Dict, trade_direction: Any, **kwargs) -> float: 
        """
        Custom stake calculation. For futures, this usually returns the configured stake amount,
        as position size is often managed by leverage.
        """
        # Always return the base stake amount, leverage will be handled by custom_entry_leverage
        return self.wallets.stake_amount

    def custom_entry_leverage(self, pair: str, current_time: datetime, current_rate: float,
                              side: str, **kwargs) -> float:
        """
        Custom entry leverage for dynamic leverage.
        This method is called prior to opening a trade.
        """
        if not self.dynamic_leverage_enable.value:
            # If dynamic leverage is disabled, return 1x leverage as Freqtrade will apply config's max_leverage
            return 1.0 

        # Get the informative 1h dataframe to calculate ATR for volatility
        # Ensure we have enough data for a valid ATR calculation
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.informative_timeframe)

        if dataframe.empty or 'ATR' not in dataframe.columns or dataframe['ATR'].iloc[-1] is np.nan:
            print(f"WARNING: ATR data not available for dynamic leverage for {pair}. Using default leverage ({self.default_leverage.value}x).")
            return float(self.default_leverage.value)

        last_1h_atr = dataframe['ATR'].iloc[-1]
        current_price = dataframe['close'].iloc[-1] # Use the current close from the 1h timeframe, or current_rate if preferred

        if current_price <= 0:
            print(f"WARNING: Current price for {pair} is zero or negative. Using default leverage ({self.default_leverage.value}x).")
            return float(self.default_leverage.value)

        atr_percent = last_1h_atr / current_price

        leverage = float(self.default_leverage.value) # Start with default

        if atr_percent > self.volatility_threshold_high.value:
            leverage = float(self.leverage_high_vol.value)
        elif atr_percent < self.volatility_threshold_low.value:
            leverage = float(self.leverage_low_vol.value)

        return leverage

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Custom stoploss logic for both LONG and SHORT positions.
        This function is called prior to all other stoploss mechanisms.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return self.stoploss # Fallback to strategy's default stoploss

        last_candle = dataframe.iloc[-1].squeeze()

        # Ensure required informative columns are present and not NaN
        required_cols_info = ["sma_200_dec_1h", "sma_200_inc_1h", "rsi_1h"]
        if not all(col in last_candle.index and pd.notna(last_candle[col]) for col in required_cols_info):
            print(f"WARNING: Missing or NaN informative indicator data for custom_stoploss for {pair}. Returning default stoploss.")
            return self.stoploss

        # Ensure required main timeframe columns are present and not NaN
        required_cols_main = ["sma_200_dec", "sma_200_inc", "ema_200", "open", "close"]
        if not all(col in last_candle.index and pd.notna(last_candle[col]) for col in required_cols_main):
            print(f"WARNING: Missing or NaN main timeframe indicator data for custom_stoploss for {pair}. Returning default stoploss.")
            return self.stoploss


        if trade.trade_direction == 'long':
            if current_profit > 0:
                return 0.99  # No stoploss if in profit (let custom_exit / trailing stop handle)
            else:
                trade_time_50 = trade.open_date_utc + timedelta(minutes=50)
                trade_time_280 = trade.open_date_utc + timedelta(minutes=280)

                # Condition set 1: After 50 minutes, check for bearish signals
                if current_time > trade_time_50:
                    if (last_candle["sma_200_dec"]) and (last_candle["sma_200_dec_1h"]): # Both SMAs declining
                        return 0.01 # Exit at 1% loss (aggressive early exit)
                    if last_candle["rsi_1h"] < 30: # 1h RSI very low, potential further downside
                        return 0.99 # Effectively no stoploss, wait for recovery or other exit
                    if last_candle["close"] < last_candle["ema_200"]: # Price below EMA200
                        if current_rate * 1.025 < last_candle["open"]: # Significant drop from open
                            return 0.01
                    if current_rate * 1.015 < last_candle["open"]: # Moderate drop from open
                            return 0.01

                # Condition set 2: After 280 minutes OR if profit falls below a specific threshold
                if (current_profit < 0) and (current_time > trade_time_280):
                    return 0.01 # Exit at 1% loss after prolonged time
                elif current_profit < self.custom_stoploss_long_1.value: # If loss reaches optimized threshold
                    if last_candle["rsi_1h"] < 30:
                        return 0.99 # Wait for recovery
                    if last_candle["close"] < last_candle["ema_200"]:
                        if current_rate * 1.025 < last_candle["open"]:
                            return 0.01
                    if current_rate * 1.015 < last_candle["open"]:
                            return 0.01
            
            return self.stoploss # Fallback to strategy's default if no custom condition met

        elif trade.trade_direction == 'short':
            if current_profit > 0:
                return 0.99 # No stoploss if in profit
            else:
                trade_time_50 = trade.open_date_utc + timedelta(minutes=50)
                trade_time_280 = trade.open_date_utc + timedelta(minutes=280)

                # Condition set 1: After 50 minutes, check for bullish signals (inverse of long)
                if current_time > trade_time_50:
                    if (last_candle["sma_200_inc"]) and (last_candle["sma_200_inc_1h"]): # Both SMAs increasing
                        return 0.01 # Exit at 1% loss (aggressive early exit)
                    if last_candle["rsi_1h"] > 70: # 1h RSI very high, potential further upside
                        return 0.99 # Effectively no stoploss, wait for recovery or other exit
                    if last_candle["close"] > last_candle["ema_200"]: # Price above EMA200
                        if current_rate * 0.975 > last_candle["open"]: # Significant rally from open (for short)
                            return 0.01
                    if current_rate * 0.985 > last_candle["open"]: # Moderate rally from open
                        return 0.01

                # Condition set 2: After 280 minutes OR if profit falls below a specific threshold
                if (current_profit < 0) and (current_time > trade_time_280):
                    return 0.01 # Exit at 1% loss after prolonged time
                elif current_profit < self.custom_stoploss_short_1.value: # If loss reaches optimized threshold
                    if last_candle["rsi_1h"] > 70:
                        return 0.99 # Wait for recovery
                    if last_candle["close"] > last_candle["ema_200"]:
                        if current_rate * 0.975 > last_candle["open"]:
                            return 0.01
                    if current_rate * 0.985 > last_candle["open"]:
                            return 0.01
            return self.stoploss # Fallback to strategy's default if no custom condition met

        return self.stoploss # Default for unexpected trade direction


    def custom_exit(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs) -> Optional[str]:
        """
        Custom exit (take profit) logic for both LONG and SHORT positions.
        This function is called to override the ROI table.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return None

        last_candle = dataframe.iloc[-1].squeeze()

        # Ensure 'rsi' and SMA trend columns are present and not NaN
        required_cols = ["rsi", "sma_200_dec", "sma_200_inc"]
        if not all(col in last_candle.index and pd.notna(last_candle[col]) for col in required_cols):
            print(f"WARNING: Missing or NaN indicator data for custom_exit for {pair}. Exiting custom exit logic.")
            return None

        if trade.trade_direction == 'long':
            # --- Long Exit Conditions ---
            # Custom ROI targets based on RSI for LONG
            if (current_profit > self.custom_roi_profit_4.value) and \
               (last_candle["rsi"] < self.custom_roi_rsi_4.value):
                return "long_roi_target_4"
            elif (current_profit > self.custom_roi_profit_3.value) and \
                 (last_candle["rsi"] < self.custom_roi_rsi_3.value):
                return "long_roi_target_3"
            elif (current_profit > self.custom_roi_profit_2.value) and \
                 (last_candle["rsi"] < self.custom_roi_rsi_2.value):
                return "long_roi_target_2"
            elif (current_profit > self.custom_roi_profit_1.value) and \
                 (last_candle["rsi"] < self.custom_roi_rsi_1.value):
                return "long_roi_target_1"
            
            # Aggressive exit for long if profit is positive but SMA200 starts declining
            elif (current_profit > 0) and \
                 (current_profit < self.custom_roi_profit_aggressive.value) and \
                 (last_candle["sma_200_dec"]):
                return "long_aggressive_exit"

        elif trade.trade_direction == 'short':
            # --- Short Exit Conditions (Inverse Logic) ---
            # Custom ROI targets based on RSI for SHORT
            # For short, higher profit means price went down, so current_profit would be positive.
            # RSI thresholds are inverted: low RSI means oversold (potential bounce up, exit short).
            if (current_profit > self.custom_roi_profit_4.value) and \
               (last_candle["rsi"] > (100 - self.custom_roi_rsi_4.value)): # Example: RSI > 40 if original was RSI < 60
                return "short_roi_target_4"
            elif (current_profit > self.custom_roi_profit_3.value) and \
                 (last_candle["rsi"] > (100 - self.custom_roi_rsi_3.value)):
                return "short_roi_target_3"
            elif (current_profit > self.custom_roi_profit_2.value) and \
                 (last_candle["rsi"] > (100 - self.custom_roi_rsi_2.value)):
                return "short_roi_target_2"
            elif (current_profit > self.custom_roi_profit_1.value) and \
                 (last_candle["rsi"] > (100 - self.custom_roi_rsi_1.value)):
                return "short_roi_target_1"

            # Aggressive exit for short if profit is positive but SMA200 is increasing
            elif (current_profit > 0) and \
                 (current_profit < self.custom_roi_profit_aggressive.value) and \
                 (last_candle["sma_200_inc"]):
                return "short_aggressive_exit"
        
        return None # No custom exit signal
