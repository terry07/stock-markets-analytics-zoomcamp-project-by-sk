# -*- coding: utf-8 -*-
import datetime
from typing import Union

import numpy as np
import pandas as pd
import pandas_datareader as pdr
import talib
import yfinance as yf


def get_date_of_previous_month() -> str:
    """Set end_date as the last date of the previous month."""

    today = pd.to_datetime('today')
    first_day_this_month = today.replace(day=1)

    end_date = (first_day_this_month - pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    return end_date

def clean_market_cap(df) -> pd.DataFrame:
    """Clean 'Market Cap' column and convert to numeric"""
    df['Market Cap'] = df['Market Cap'].str.replace('$', '').str.replace('B', '').str.replace('M', '').str.replace(',', '')
    df['Market Cap'] = pd.to_numeric(df['Market Cap'], errors='coerce')

    return df


def features_based_on_price(df, ticker) -> pd.DataFrame:
    """Download daily prices for each provided and valid ticker,
      and calculate several related features.

      The final dataset is resampled on monthly basis (month-end).
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


    # round data to 6 decimal places
    df_resampled = df_resampled.round(6)


    # add typical metadata for completeness
    df_resampled['year'] = df_resampled['Date'].dt.year
    df_resampled['month'] = df_resampled['Date'].dt.month
    df_resampled['ticker'] = ticker

    # drop rows with NaN values and reset index
    df_resampled = df_resampled.dropna().reset_index(drop=True)
    print(f"Final data shape for {ticker}: ", df_resampled.shape)

    return df_resampled


def safe_get_latest(bs_df, row_name) -> Union[float, int, str]:
    """Safely get the latest value from a balance sheet DataFrame."""
    if bs_df is not None and row_name in bs_df.index:

        series = bs_df.loc[row_name].dropna()
        if not series.empty:
            return series.iloc[0]

    return np.nan

def features_based_on_fundamentals(ticker, end_date) -> dict:
    """Download quarterly fundamentals for each provided and valid ticker,
      and calculate several related features.

      The final dataset is resampled on monthly basis (month-end).
    """
    tk = yf.Ticker(ticker)

    # extract fundamental data based on provided ticket
    actions = tk.get_actions()
    bs = tk.get_balance_sheet()
    calendar = tk.get_calendar()
    cf = tk.get_cashflow()
    holders = tk.get_institutional_holders()
    info = tk.get_info() or {}
    recommendations = tk.get_recommendations()
    sustainability = tk.get_sustainability()


    # get the latest available values safely before BS-based calculations
    total_assets = safe_get_latest(bs, 'TotalAssets')
    total_liabilities = safe_get_latest(bs, 'TotalLiabilitiesNetMinorityInterest')
    total_equity = safe_get_latest(bs, 'StockholdersEquity')
    long_term_debt = safe_get_latest(bs, 'LongTermDebt')
    current_assets = safe_get_latest(bs, 'CurrentAssets')
    current_liabilities = safe_get_latest(bs, 'CurrentLiabilities')
    cash_equivalents = safe_get_latest(bs, 'CashAndCashEquivalents')
    retained_earnings = safe_get_latest(bs, 'RetainedEarnings')
    working_capital = safe_get_latest(bs, 'WorkingCapital')


    # Balance Sheet parameters
    debt_to_equity = total_liabilities / total_equity if total_equity else np.nan
    current_ratio = current_assets / current_liabilities if current_liabilities else np.nan
    cash_ratio = cash_equivalents / current_liabilities if current_liabilities else np.nan
    working_capital_ratio = working_capital / total_assets if total_assets else np.nan
    retained_earnings_to_assets = retained_earnings / total_assets if total_assets else np.nan
    long_term_debt_to_equity = long_term_debt / total_equity if total_equity else np.nan
    op_cf = cf.loc['OperatingCashFlow'].iloc[0] if cf is not None and 'OperatingCashFlow' in cf.index else None
    revenue = info.get('totalRevenue')
    operating_cf_margin = op_cf / revenue if op_cf and revenue else np.nan
    free_cf = cf.loc['FreeCashFlow'].iloc[0] if cf is not None and 'FreeCashFlow' in cf.index else None
    free_cf_margin = free_cf / revenue if free_cf and revenue else np.nan


    # Institutional holders
    pct_held = holders['pctHeld'].mean() if holders is not None and not holders.empty else np.nan
    inst_count = len(holders) if holders is not None else np.nan


    # ESG scores
    esg_env = sustainability.loc['environmentScore'].iloc[0] if sustainability is not None and 'environmentScore' in sustainability.index else np.nan
    esg_soc = sustainability.loc['socialScore'].iloc[0] if sustainability is not None and 'socialScore' in sustainability.index else np.nan
    esg_gov = sustainability.loc['governanceScore'].iloc[0] if sustainability is not None and 'governanceScore' in sustainability.index else np.nan


    # Recommendations up to 3 months back
    recent_reco = 0
    if recommendations is not None and not recommendations.empty:
        recent_months = recommendations[recommendations['period'].isin(['0m', '-1m', '-2m', '-3m'])]
        recent_reco = recent_months[['strongBuy', 'buy', 'hold', 'sell', 'strongSell']].sum().sum()
        strong_to_total_ratio = recent_months['strongBuy'].sum() / recent_reco if recent_reco else np.nan


    # Days to next earnings
    days_to_earnings = np.nan
    if calendar and isinstance(calendar, dict) and 'Earnings Date' in calendar:
        earnings_dates = calendar['Earnings Date']
        if isinstance(earnings_dates, list) and len(earnings_dates) > 0:
            earnings_date = earnings_dates[0]
            if isinstance(earnings_date, (datetime.date, datetime.datetime)):
                earnings_datetime = pd.to_datetime(earnings_date)
                days_to_earnings = (earnings_datetime - pd.to_datetime(end_date)).days


    # Dividend stability: std of yearly dividend sums / mean dividend sums
    dividend_stability = np.nan
    if actions is not None and 'Dividends' in actions.columns:
        yearly_divs = actions['Dividends'].resample('YE').sum()
        if len(yearly_divs) > 1 and yearly_divs.mean() != 0:
            dividend_stability = yearly_divs.std() / yearly_divs.mean()

    fund_feats = {
        'marketCap': info.get('marketCap', np.nan),
        'beta': info.get('beta', np.nan),
        'trailingPE': info.get('trailingPE', np.nan),
        'forwardPE': info.get('forwardPE', np.nan),
        'trailing_PEG': info.get('trailingPegRatio', np.nan),
        'priceToBook': info.get('priceToBook', np.nan),
        'dividendYield': info.get('dividendYield', np.nan),
        'debt_to_equity': debt_to_equity,
        'debt_to_equity': debt_to_equity,
        'current_ratio': current_ratio,
        'cash_ratio': cash_ratio,
        'working_capital': working_capital,
        'working_capital_ratio': working_capital_ratio,
        'retained_earnings_to_assets': retained_earnings_to_assets,
        'long_term_debt_to_equity': long_term_debt_to_equity,
        'operating_cf_margin': operating_cf_margin,
        'free_cf_margin': free_cf_margin,
        'pct_held_by_inst': pct_held,
        'institutional_holders_count': inst_count,
        'esg_env': esg_env,
        'esg_soc': esg_soc,
        'esg_gov': esg_gov,
        'recent_rating_changes': recent_reco,
        'strong_to_total_reco_ratio': strong_to_total_ratio,
        'days_to_next_earnings': days_to_earnings,
        'dividend_stability': dividend_stability,
        'companyName': info.get('shortName', np.nan),
        'sector': info.get('sector'),
        'industry': info.get('industry'),
        'country': info.get('country'),
        'fullTimeEmployees': info.get('fullTimeEmployees', np.nan),
    }

    # round float features to 6 decimal places, keep NaNs and categorical variables intact
    fund_feats_float = {k: np.round(v,6) if isinstance(v, (int, float, np.integer, np.floating)) and not pd.isnull(v)
                        else v for k, v in fund_feats.items()}

    # count missing values
    missing_count = sum(pd.isnull(v) for v in fund_feats_float.values())
    print(f"Number of missing values in fund_feats_float: {missing_count}")

    return fund_feats_float


def get_talib_pattern_indicators(df) -> pd.DataFrame:
    """Computes a comprehensive set of candlestick pattern recognition indicators using TA-Lib for the given DataFrame.

    This function applies all available TA-Lib candlestick pattern functions to the input DataFrame, which must contain
    at least the columns: 'Open', 'High', 'Low', 'Close', 'Date', and 'Ticker'. It returns a new DataFrame with the
    original 'Date' and 'Ticker' columns, along with one column for each candlestick pattern indicator.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing at least the columns 'Open', 'High', 'Low', 'Close', 'Date', and 'Ticker'.

    Returns:
        pd.DataFrame: A DataFrame with columns for 'Date', 'Ticker', and all TA-Lib candlestick pattern indicators.
                      The 'Date' column is converted to pandas datetime type.

    Useful link: https://github.com/TA-Lib/ta-lib-python/blob/master/docs/func_groups/pattern_recognition.md

    Nice article about candles (pattern recognition):
      https://medium.com/analytics-vidhya/recognizing-over-50-candlestick-patterns-with-python-4f02a1822cb5
    """
    # CDL2CROWS - Two Crows
    talib_cdl2crows = talib.CDL2CROWS(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDL3BLACKCROWS - Three Black Crows
    talib_cdl3blackrows = talib.CDL3BLACKCROWS(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDL3INSIDE - Three Inside Up/Down
    talib_cdl3inside = talib.CDL3INSIDE(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDL3LINESTRIKE - Three-Line Strike
    talib_cdl3linestrike = talib.CDL3LINESTRIKE(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDL3OUTSIDE - Three Outside Up/Down
    talib_cdl3outside = talib.CDL3OUTSIDE(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDL3STARSINSOUTH - Three Stars In The South
    talib_cdl3starsinsouth = talib.CDL3STARSINSOUTH(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDL3WHITESOLDIERS - Three Advancing White Soldiers
    talib_cdl3whitesoldiers = talib.CDL3WHITESOLDIERS(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLABANDONEDBABY - Abandoned Baby
    talib_cdlabandonedbaby = talib.CDLABANDONEDBABY(
            df.Open.values, df.High.values, df.Low.values, df.Close.values, penetration=0)
    # CDLADVANCEBLOCK - Advance Block
    talib_cdladvancedblock = talib.CDLADVANCEBLOCK(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLBELTHOLD - Belt-hold
    talib_cdlbelthold = talib.CDLBELTHOLD(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLBREAKAWAY - Breakaway
    talib_cdlbreakaway = talib.CDLBREAKAWAY(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLCLOSINGMARUBOZU - Closing Marubozu
    talib_cdlclosingmarubozu = talib.CDLCLOSINGMARUBOZU(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLCONCEALBABYSWALL - Concealing Baby Swallow
    talib_cdlconcealbabyswall = talib.CDLCONCEALBABYSWALL(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLCOUNTERATTACK - Counterattack
    talib_cdlcounterattack = talib.CDLCOUNTERATTACK(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLDARKCLOUDCOVER - Dark Cloud Cover
    talib_cdldarkcloudcover = talib.CDLDARKCLOUDCOVER(
            df.Open.values, df.High.values, df.Low.values, df.Close.values, penetration=0)
    # CDLDOJI - Doji
    talib_cdldoji = talib.CDLDOJI(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLDOJISTAR - Doji Star
    talib_cdldojistar = talib.CDLDOJISTAR(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLDRAGONFLYDOJI - Dragonfly Doji
    talib_cdldragonflydoji = talib.CDLDRAGONFLYDOJI(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLENGULFING - Engulfing Pattern
    talib_cdlengulfing = talib.CDLENGULFING(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)

    # CDLEVENINGDOJISTAR - Evening Doji Star
    talib_cdleveningdojistar = talib.CDLEVENINGDOJISTAR(
            df.Open.values, df.High.values, df.Low.values, df.Close.values, penetration=0)
    # CDLEVENINGSTAR - Evening Star
    talib_cdleveningstar = talib.CDLEVENINGSTAR(
            df.Open.values, df.High.values, df.Low.values, df.Close.values, penetration=0)
    # CDLGAPSIDESIDEWHITE - Up/Down-gap side-by-side white lines
    talib_cdlgapsidesidewhite = talib.CDLGAPSIDESIDEWHITE(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLGRAVESTONEDOJI - Gravestone Doji
    talib_cdlgravestonedoji = talib.CDLGRAVESTONEDOJI(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLHAMMER - Hammer
    talib_cdlhammer = talib.CDLHAMMER(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLHANGINGMAN - Hanging Man
    talib_cdlhangingman = talib.CDLHANGINGMAN(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLHARAMI - Harami Pattern
    talib_cdlharami = talib.CDLHARAMI(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLHARAMICROSS - Harami Cross Pattern
    talib_cdlharamicross = talib.CDLHARAMICROSS(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLHIGHWAVE - High-Wave Candle
    talib_cdlhighwave = talib.CDLHIGHWAVE(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLHIKKAKE - Hikkake Pattern
    talib_cdlhikkake = talib.CDLHIKKAKE(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLHIKKAKEMOD - Modified Hikkake Pattern
    talib_cdlhikkakemod = talib.CDLHIKKAKEMOD(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)

    # CDLHOMINGPIGEON - Homing Pigeon
    talib_cdlhomingpigeon = talib.CDLHOMINGPIGEON(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLIDENTICAL3CROWS - Identical Three Crows
    talib_cdlidentical3crows = talib.CDLIDENTICAL3CROWS(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLINNECK - In-Neck Pattern
    talib_cdlinneck = talib.CDLINNECK(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLINVERTEDHAMMER - Inverted Hammer
    talib_cdlinvertedhammer = talib.CDLINVERTEDHAMMER(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLKICKING - Kicking
    talib_cdlkicking = talib.CDLKICKING(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLKICKINGBYLENGTH - Kicking - bull/bear determined by the longer marubozu
    talib_cdlkickingbylength = talib.CDLKICKINGBYLENGTH(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLLADDERBOTTOM - Ladder Bottom
    talib_cdlladderbottom = talib.CDLLADDERBOTTOM(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLLONGLEGGEDDOJI - Long Legged Doji
    talib_cdllongleggeddoji = talib.CDLLONGLEGGEDDOJI(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLLONGLINE - Long Line Candle
    talib_cdllongline = talib.CDLLONGLINE(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLMARUBOZU - Marubozu
    talib_cdlmarubozu = talib.CDLMARUBOZU(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLMATCHINGLOW - Matching Low
    talib_cdlmatchinglow = talib.CDLMATCHINGLOW(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)

    # CDLMATHOLD - Mat Hold
    talib_cdlmathold = talib.CDLMATHOLD(
            df.Open.values, df.High.values, df.Low.values, df.Close.values, penetration=0)
    # CDLMORNINGDOJISTAR - Morning Doji Star
    talib_cdlmorningdojistar = talib.CDLMORNINGDOJISTAR(
            df.Open.values, df.High.values, df.Low.values, df.Close.values, penetration=0)
    # CDLMORNINGSTAR - Morning Star
    talib_cdlmorningstar = talib.CDLMORNINGSTAR(
            df.Open.values, df.High.values, df.Low.values, df.Close.values, penetration=0)
    # CDLONNECK - On-Neck Pattern
    talib_cdlonneck = talib.CDLONNECK(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLPIERCING - Piercing Pattern
    talib_cdlpiercing = talib.CDLPIERCING(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLRICKSHAWMAN - Rickshaw Man
    talib_cdlrickshawman = talib.CDLRICKSHAWMAN(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLRISEFALL3METHODS - Rising/Falling Three Methods
    talib_cdlrisefall3methods = talib.CDLRISEFALL3METHODS(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLSEPARATINGLINES - Separating Lines
    talib_cdlseparatinglines = talib.CDLSEPARATINGLINES(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLSHOOTINGSTAR - Shooting Star
    talib_cdlshootingstar = talib.CDLSHOOTINGSTAR(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLSHORTLINE - Short Line Candle
    talib_cdlshortline = talib.CDLSHORTLINE(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLSPINNINGTOP - Spinning Top
    talib_cdlspinningtop = talib.CDLSPINNINGTOP(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)

    # CDLSTALLEDPATTERN - Stalled Pattern
    talib_cdlstalledpattern = talib.CDLSTALLEDPATTERN(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLSTICKSANDWICH - Stick Sandwich
    talib_cdlsticksandwich = talib.CDLSTICKSANDWICH(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLTAKURI - Takuri (Dragonfly Doji with very long lower shadow)
    talib_cdltakuru = talib.CDLTAKURI(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLTASUKIGAP - Tasuki Gap
    talib_cdltasukigap = talib.CDLTASUKIGAP(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLTHRUSTING - Thrusting Pattern
    talib_cdlthrusting = talib.CDLTHRUSTING(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLTRISTAR - Tristar Pattern
    talib_cdltristar = talib.CDLTRISTAR(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLUNIQUE3RIVER - Unique 3 River
    talib_cdlunique3river = talib.CDLUNIQUE3RIVER(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLUPSIDEGAP2CROWS - Upside Gap Two Crows
    talib_cdlupsidegap2crows = talib.CDLUPSIDEGAP2CROWS(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CDLXSIDEGAP3METHODS - Upside/Downside Gap Three Methods
    talib_cdlxsidegap3methods = talib.CDLXSIDEGAP3METHODS(
            df.Open.values, df.High.values, df.Low.values, df.Close.values)

    pattern_indicators_df = pd.DataFrame(
            {'Date': df.Date.values,
             'Ticker': df.Ticker,
             # TA-Lib Pattern Recognition indicators
             'cdl2crows': talib_cdl2crows,
             'cdl3blackrows': talib_cdl3blackrows,
             'cdl3inside': talib_cdl3inside,
             'cdl3linestrike': talib_cdl3linestrike,
             'cdl3outside': talib_cdl3outside,
             'cdl3starsinsouth': talib_cdl3starsinsouth,
             'cdl3whitesoldiers': talib_cdl3whitesoldiers,
             'cdlabandonedbaby': talib_cdlabandonedbaby,
             'cdladvancedblock': talib_cdladvancedblock,
             'cdlbelthold': talib_cdlbelthold,
             'cdlbreakaway': talib_cdlbreakaway,
             'cdlclosingmarubozu': talib_cdlclosingmarubozu,
             'cdlconcealbabyswall': talib_cdlconcealbabyswall,
             'cdlcounterattack': talib_cdlcounterattack,
             'cdldarkcloudcover': talib_cdldarkcloudcover,
             'cdldoji': talib_cdldoji,
             'cdldojistar': talib_cdldojistar,
             'cdldragonflydoji': talib_cdldragonflydoji,
             'cdlengulfing': talib_cdlengulfing,
             'cdleveningdojistar': talib_cdleveningdojistar,
             'cdleveningstar': talib_cdleveningstar,
             'cdlgapsidesidewhite': talib_cdlgapsidesidewhite,
             'cdlgravestonedoji': talib_cdlgravestonedoji,
             'cdlhammer': talib_cdlhammer,
             'cdlhangingman': talib_cdlhangingman,
             'cdlharami': talib_cdlharami,
             'cdlharamicross': talib_cdlharamicross,
             'cdlhighwave': talib_cdlhighwave,
             'cdlhikkake': talib_cdlhikkake,
             'cdlhikkakemod': talib_cdlhikkakemod,
             'cdlhomingpigeon': talib_cdlhomingpigeon,
             'cdlidentical3crows': talib_cdlidentical3crows,
             'cdlinneck': talib_cdlinneck,
             'cdlinvertedhammer': talib_cdlinvertedhammer,
             'cdlkicking': talib_cdlkicking,
             'cdlkickingbylength': talib_cdlkickingbylength,
             'cdlladderbottom': talib_cdlladderbottom,
             'cdllongleggeddoji': talib_cdllongleggeddoji,
             'cdllongline': talib_cdllongline,
             'cdlmarubozu': talib_cdlmarubozu,
             'cdlmatchinglow': talib_cdlmatchinglow,
             'cdlmathold': talib_cdlmathold,
             'cdlmorningdojistar': talib_cdlmorningdojistar,
             'cdlmorningstar': talib_cdlmorningstar,
             'cdlonneck': talib_cdlonneck,
             'cdlpiercing': talib_cdlpiercing,
             'cdlrickshawman': talib_cdlrickshawman,
             'cdlrisefall3methods': talib_cdlrisefall3methods,
             'cdlseparatinglines': talib_cdlseparatinglines,
             'cdlshootingstar': talib_cdlshootingstar,
             'cdlshortline': talib_cdlshortline,
             'cdlspinningtop': talib_cdlspinningtop,
             'cdlstalledpattern': talib_cdlstalledpattern,
             'cdlsticksandwich': talib_cdlsticksandwich,
             'cdltakuru': talib_cdltakuru,
             'cdltasukigap': talib_cdltasukigap,
             'cdlthrusting': talib_cdlthrusting,
             'cdltristar': talib_cdltristar,
             'cdlunique3river': talib_cdlunique3river,
             'cdlupsidegap2crows': talib_cdlupsidegap2crows,
             'cdlxsidegap3methods': talib_cdlxsidegap3methods
             }
        )

    # Need a proper date type
    pattern_indicators_df['Date'] = pd.to_datetime(
            pattern_indicators_df['Date'])

    return pattern_indicators_df


def get_talib_momentum_indicators(df) -> pd.DataFrame:
    """Calculate various momentum indicators using TA-Lib for a given DataFrame of stock data.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least the following columns:
        - 'Date': Date of the record
        - 'Ticker': Stock ticker symbol
        - 'Open': Opening price
        - 'High': Highest price
        - 'Low': Lowest price
        - 'Close': Closing price
        - 'Volume': Trading volume (required for some indicators, e.g., MFI)

    Returns
    -------
    pd.DataFrame
        DataFrame containing the original 'Date' and 'Ticker' columns, along with the following TA-Lib momentum indicators:
        - adx: Average Directional Movement Index
        - adxr: Average Directional Movement Index Rating
        - apo: Absolute Price Oscillator
        - aroon_1: Aroon Up
        - aroon_2: Aroon Down
        - aroonosc: Aroon Oscillator
        - bop: Balance of Power
        - cci: Commodity Channel Index
        - cmo: Chande Momentum Oscillator
        - dx: Directional Movement Index
        - macd: MACD line
        - macdsignal: MACD signal line
        - macdhist: MACD histogram
        - macd_ext: MACD (extended version)
        - macdsignal_ext: MACD signal (extended)
        - macdhist_ext: MACD histogram (extended)
        - macd_fix: MACD (fixed version)
        - macdsignal_fix: MACD signal (fixed)
        - macdhist_fix: MACD histogram (fixed)
        - minus_di: Minus Directional Indicator
        - mom: Momentum
        - plus_di: Plus Directional Indicator
        - dm: Plus Directional Movement
        - ppo: Percentage Price Oscillator
        - roc: Rate of Change
        - rocp: Rate of Change Percentage
        - rocr: Rate of Change Ratio
        - rocr100: Rate of Change Ratio (100 scale)
        - rsi: Relative Strength Index
        - slowk: Stochastic %K (slow)
        - slowd: Stochastic %D (slow)
        - fastk: Stochastic %K (fast)
        - fastd: Stochastic %D (fast)
        - fastk_rsi: Stochastic RSI %K
        - fastd_rsi: Stochastic RSI %D
        - trix: 1-day ROC of a Triple Smooth EMA
        - ultosc: Ultimate Oscillator
        - willr: Williams' %R

    Notes
    -----
    - The function assumes the input DataFrame columns are named as specified.
    - Some indicators may return NaN for initial periods due to lookback windows.
    - The Money Flow Index (MFI) is commented out and not included in the output.
    """
    momentum_df = None
    # ADX - Average Directional Movement Index
    talib_momentum_adx = talib.ADX(df.High.values, df.Low.values, df.Close.values, timeperiod=14)
    # ADXR - Average Directional Movement Index Rating
    talib_momentum_adxr = talib.ADXR(df.High.values, df.Low.values, df.Close.values, timeperiod=14 )
    # APO - Absolute Price Oscillator
    talib_momentum_apo = talib.APO(df.Close.values, fastperiod=12, slowperiod=26, matype=0 )
    # AROON - Aroon
    talib_momentum_aroon = talib.AROON(df.High.values, df.Low.values, timeperiod=14 )
    # talib_momentum_aroon[0].size
    # talib_momentum_aroon[1].size
    # AROONOSC - Aroon Oscillator
    talib_momentum_aroonosc = talib.AROONOSC(df.High.values, df.Low.values, timeperiod=14)
    # BOP - Balance of Power
    # https://school.stockcharts.com/doku.php?id=technical_indicators:balance_of_power
    #calculate open prices as shifted closed prices from the prev day
    # open = df.Last.shift(1)
    talib_momentum_bop = talib.BOP(df.Open.values, df.High.values, df.Low.values, df.Close.values)
    # CCI - Commodity Channel Index
    talib_momentum_cci = talib.CCI(df.High.values, df.Low.values, df.Close.values, timeperiod=14)
    # CMO - Chande Momentum Oscillator
    talib_momentum_cmo = talib.CMO(df.Close.values, timeperiod=14)
    # DX - Directional Movement Index
    talib_momentum_dx = talib.DX(df.High.values, df.Low.values, df.Close.values, timeperiod=14)
    # MACD - Moving Average Convergence/Divergence
    talib_momentum_macd, talib_momentum_macdsignal, talib_momentum_macdhist = talib.MACD(df.Close.values, fastperiod=12, \
                                                                                        slowperiod=26, signalperiod=9)
    # MACDEXT - MACD with controllable MA type
    talib_momentum_macd_ext, talib_momentum_macdsignal_ext, talib_momentum_macdhist_ext = talib.MACDEXT(df.Close.values, \
                                                                                                      fastperiod=12, \
                                                                                                      fastmatype=0, \
                                                                                                      slowperiod=26, \
                                                                                                      slowmatype=0, \
                                                                                                      signalperiod=9, \
                                                                                                    signalmatype=0)
    # MACDFIX - Moving Average Convergence/Divergence Fix 12/26
    talib_momentum_macd_fix, talib_momentum_macdsignal_fix, talib_momentum_macdhist_fix = talib.MACDFIX(df.Close.values, \
                                                                                                        signalperiod=9)
    # MFI - Money Flow Index
    #talib_momentum_mfi = talib.MFI(df.High.values, df.Low.values, df.Close.values, df.Volume.values, timeperiod=14)
    # MINUS_DI - Minus Directional Indicator
    talib_momentum_minus_di = talib.MINUS_DM(df.High.values, df.Low.values, timeperiod=14)
    # MOM - Momentum
    talib_momentum_mom = talib.MOM(df.Close.values, timeperiod=10)
    # PLUS_DI - Plus Directional Indicator
    talib_momentum_plus_di = talib.PLUS_DI(df.High.values, df.Low.values, df.Close.values, timeperiod=14)
    # PLUS_DM - Plus Directional Movement
    talib_momentum_plus_dm = talib.PLUS_DM(df.High.values, df.Low.values, timeperiod=14)
    # PPO - Percentage Price Oscillator
    talib_momentum_ppo = talib.PPO(df.Close.values, fastperiod=12, slowperiod=26, matype=0)
    # ROC - Rate of change : ((price/prevPrice)-1)*100
    talib_momentum_roc = talib.ROC(df.Close.values, timeperiod=10)
    # ROCP - Rate of change Percentage: (price-prevPrice)/prevPrice
    talib_momentum_rocp = talib.ROCP(df.Close.values, timeperiod=10)
    # ROCR - Rate of change ratio: (price/prevPrice)
    talib_momentum_rocr = talib.ROCR(df.Close.values, timeperiod=10)
    # ROCR100 - Rate of change ratio 100 scale: (price/prevPrice)*100
    talib_momentum_rocr100 = talib.ROCR100(df.Close.values, timeperiod=10)
    # RSI - Relative Strength Index
    talib_momentum_rsi = talib.RSI(df.Close.values, timeperiod=14)
    # STOCH - Stochastic
    talib_momentum_slowk, talib_momentum_slowd = talib.STOCH(df.High.values, df.Low.values, df.Close.values, \
                                                            fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    # STOCHF - Stochastic Fast
    talib_momentum_fastk, talib_momentum_fastd = talib.STOCHF(df.High.values, df.Low.values, df.Close.values, \
                                                              fastk_period=5, fastd_period=3, fastd_matype=0)
    # STOCHRSI - Stochastic Relative Strength Index
    talib_momentum_fastk_rsi, talib_momentum_fastd_rsi = talib.STOCHRSI(df.Close.values, timeperiod=14, \
                                                                        fastk_period=5, fastd_period=3, fastd_matype=0)
    # TRIX - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
    talib_momentum_trix = talib.TRIX(df.Close.values, timeperiod=30)
    # ULTOSC - Ultimate Oscillator
    talib_momentum_ultosc = talib.ULTOSC(df.High.values, df.Low.values, df.Close.values, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    # WILLR - Williams' %R
    talib_momentum_willr = talib.WILLR(df.High.values, df.Low.values, df.Close.values, timeperiod=14)

    momentum_df =   pd.DataFrame(
      {
        # assume here multi-index <dateTime, ticker>
        # 'datetime': df.index.get_level_values(0),
        # 'ticker': df.index.get_level_values(1) ,

        # old way with separate columns
        'Date': df.Date.values,
        'Ticker': df.Ticker,

        'adx': talib_momentum_adx,
        'adxr': talib_momentum_adxr,
        'apo': talib_momentum_apo,
        'aroon_1': talib_momentum_aroon[0] ,
        'aroon_2': talib_momentum_aroon[1],
        'aroonosc': talib_momentum_aroonosc,
        'bop': talib_momentum_bop,
        'cci': talib_momentum_cci,
        'cmo': talib_momentum_cmo,
        'dx': talib_momentum_dx,
        'macd': talib_momentum_macd,
        'macdsignal': talib_momentum_macdsignal,
        'macdhist': talib_momentum_macdhist,
        'macd_ext': talib_momentum_macd_ext,
        'macdsignal_ext': talib_momentum_macdsignal_ext,
        'macdhist_ext': talib_momentum_macdhist_ext,
        'macd_fix': talib_momentum_macd_fix,
        'macdsignal_fix': talib_momentum_macdsignal_fix,
        'macdhist_fix': talib_momentum_macdhist_fix,
        #'mfi': talib_momentum_mfi,
        'minus_di': talib_momentum_minus_di,
        'mom': talib_momentum_mom,
        'plus_di': talib_momentum_plus_di,
        'dm': talib_momentum_plus_dm,
        'ppo': talib_momentum_ppo,
        'roc': talib_momentum_roc,
        'rocp': talib_momentum_rocp,
        'rocr': talib_momentum_rocr,
        'rocr100': talib_momentum_rocr100,
        'rsi': talib_momentum_rsi,
        'slowk': talib_momentum_slowk,
        'slowd': talib_momentum_slowd,
        'fastk': talib_momentum_fastk,
        'fastd': talib_momentum_fastd,
        'fastk_rsi': talib_momentum_fastk_rsi,
        'fastd_rsi': talib_momentum_fastd_rsi,
        'trix': talib_momentum_trix,
        'ultosc': talib_momentum_ultosc,
        'willr': talib_momentum_willr,
      }
    )
    return momentum_df


def get_macro_market_data(fred_mapping, start_date, end_date) -> Union[pd.DataFrame, dict]:
    """Fetches major market indices, ETFs, and forex rates.

       Most representative examlples are S&P 500, VIX, DAX, sector ETFs, and USD indices.

    Returns
    -------
    pd.DataFrame or dict
        Real-time or historical market data for macroeconomic tickers.
    """
    print(f"Retrieve fred fearures for the selected period...\n")
    print("-"*24, "\n")

    dataset_fred = dict()

    for name, code in fred_mapping.items():

        try:

            df = pdr.DataReader(code, "fred", start=start_date, end=end_date)
            if df.empty:
                print(f"[FRED] No data for {name} ({code}), skipping.")

            else:

                # Rename column to friendly name
                df = df.rename(columns={code: name})

            dataset_fred[name] = df

        except Exception as e:
            print(f"[FRED] Error for {name} ({code}): {e}")
            continue

    if not dataset_fred:
        print("No FRED data retrieved.")
        return pd.DataFrame()

    dataset_fred = pd.concat(dataset_fred.values(), axis=1).resample('ME').last().fillna(method='ffill')

    return dataset_fred
