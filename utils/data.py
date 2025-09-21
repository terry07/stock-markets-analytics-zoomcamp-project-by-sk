# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import yfinance as yf


def get_date_of_previous_month():
    """Set end_date as the last date of the previous month."""

    today = pd.to_datetime('today')
    first_day_this_month = today.replace(day=1)

    end_date = (first_day_this_month - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    return end_date

def clean_market_cap(df):
    """Clean 'Market Cap' column and convert to numeric"""
    df['Market Cap'] = df['Market Cap'].str.replace('$', '').str.replace('B', '').str.replace('M', '').str.replace(',', '')
    df['Market Cap'] = pd.to_numeric(df['Market Cap'], errors='coerce')
    return df


def features_based_on_price(df, ticker):
    """Download daily prices for each provided and valid ticker,
      and calculate several related features.

      The final dataset is resample on monthly basis (month-end).
    """

    # Set `Adjusted Close`  in place of `Close`, if available
    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    if price_col == 'Close' and 'Adj Close' not in df.columns:
        print(f"'Adj Close' not present in columns for {ticker}."
                f"Using 'Close' instead.")

    df.rename(columns={price_col: 'Close'}, inplace=True)


    # typical messages
    print("Data columns: ", df.columns)
    print("Data index type: ", type(df.index))
    print("Data shape: ", df.shape)


    # logarithm of `Volume` (avoid log(0) by replacing 0 with NaN)
    df['ln_volume'] = np.log(df['Volume'].replace(0, np.nan))


    # growth rates calculated on different time intervals (5 days equals to 1 week)
    df['daily_growth'] = df['Close'].pct_change()
    df['weekly_growth'] = df['Close'].pct_change(5)
    df['biweekly_growth'] = df['Close'].pct_change(10)
    df['monthly_growth'] = df['Close'].pct_change(21)


    # relative spreads (scaled by Close price)
    df['rel_spread_oc'] = (df['Open'] - df['Close']) / df['Close']
    df['rel_spread_hl'] = (df['High'] - df['Low']) / df['Close']


    # statistical indicators of rolling volatility for short-term risk
    pct_change = df['Close'].pct_change()
    time_interval = 10  # 10 days
    df['vol_10d_mean'] = pct_change.rolling(time_interval).mean()
    df['vol_10d_std'] = pct_change.rolling(time_interval).std()
    df['vol_10d_min'] = pct_change.rolling(time_interval).min()
    df['vol_10d_max'] = pct_change.rolling(time_interval).max()


    # process of resampling to month-end frequency
    print("Resampling to month-end frequency... \n")

    agg_funcs = {
        'Close': 'last',
        'ln_volume': ['min', 'median', 'max'],
        'daily_growth': ['min', 'median', 'max'],
        'weekly_growth': ['min', 'median', 'max'],
        'biweekly_growth': ['min', 'median', 'max'],
        'monthly_growth': ['min', 'median', 'max'],
        'rel_spread_oc': 'median' if 'rel_spread_oc' in df else 'first',
        'rel_spread_hl': 'median' if 'rel_spread_hl' in df else 'first',
        'vol_10d_mean': 'median' if 'vol_10d_mean' in df else 'first',
        'vol_10d_std': 'median' if 'vol_10d_std' in df else 'first',
        'vol_10d_min': 'median' if 'vol_10d_min' in df else 'first',
        'vol_10d_max': 'median' if 'vol_10d_max' in df else 'first',
    }
    df_resampled = df.resample('ME').agg(agg_funcs)


    # flatten MultiIndex columns if present
    df_resampled.columns = ['_'.join(col) if isinstance(col, tuple)
                            else col for col in df_resampled.columns]
    print("Resampling done. The new data shape is ", df_resampled.shape)


    # convert index to datetime, and remove timezone
    df_resampled.index = pd.to_datetime(df_resampled.index).tz_localize(None)
    df_resampled.reset_index(inplace=True)
    df_resampled.rename(columns={'index': 'Date'}, inplace=True)
    df_resampled.rename(columns={'Close_last': 'Close'}, inplace=True)


    # calculation of historical returns on monthly scale (various lags)
    df_resampled['return_1m'] = df_resampled['Close'].pct_change(1)
    df_resampled['return_2m'] = df_resampled['Close'].pct_change(2)
    df_resampled['return_3m'] = df_resampled['Close'].pct_change(3)


    # volatility and momentum on monthly scale (3-month window)
    close_pct_change = df_resampled['Close'].pct_change()
    df_resampled['vol_3m'] = close_pct_change.rolling(window=3).std()
    df_resampled['momentum_3m'] = df_resampled['return_3m'] / df_resampled['vol_3m']


    # moving averages on monthly scale (various lags)
    df_resampled['mma_3'] = df_resampled['Close'].rolling(window=3).mean()
    df_resampled['mma_6'] = df_resampled['Close'].rolling(window=6).mean()

    # scale moving averages by current Close price
    df_resampled['mma_3_scaled'] = df_resampled['mma_3'] / df_resampled['Close']
    df_resampled['mma_6_scaled'] = df_resampled['mma_6'] / df_resampled['Close']

    # drop unscaled moving averages
    df_resampled.drop(columns=['mma_3', 'mma_6'], inplace=True)


    # round data to 4 decimal places
    df_resampled = df_resampled.round(4)


    # add typical metadata for completeness
    df_resampled['year'] = df_resampled['Date'].dt.year
    df_resampled['month'] = df_resampled['Date'].dt.month
    df_resampled['ticker'] = ticker

    # drop rows with NaN values and reset index
    df_resampled = df_resampled.dropna().reset_index(drop=True)
    print(f"Final data shape for {ticker}: ", df_resampled.shape)

    return df_resampled
