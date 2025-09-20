# -*- coding: utf-8 -*-
import pandas as pd


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
