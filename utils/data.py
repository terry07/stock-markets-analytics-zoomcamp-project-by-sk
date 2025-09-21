# -*- coding: utf-8 -*-
import datetime
from typing import Union

import numpy as np
import pandas as pd
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

    # round float features to 4 decimal places, keep NaNs and categorical variables intact
    fund_feats_float = {k: np.round(v,4) if isinstance(v, (int, float, np.integer, np.floating)) and not pd.isnull(v)
                        else v for k, v in fund_feats.items()}

    # count missing values
    missing_count = sum(pd.isnull(v) for v in fund_feats_float.values())
    print(f"Number of missing values in fund_feats_float: {missing_count}")

    return fund_feats_float
