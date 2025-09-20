# -*- coding: utf-8 -*-
import pandas as pd


def clean_market_cap(df):
    """Clean 'Market Cap' column and convert to numeric"""
    df['Market Cap'] = df['Market Cap'].str.replace('$', '').str.replace('B', '').str.replace('M', '').str.replace(',', '')
    df['Market Cap'] = pd.to_numeric(df['Market Cap'], errors='coerce')
    return df
