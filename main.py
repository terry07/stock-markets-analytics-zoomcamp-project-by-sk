#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stock Market Analytics - Main Pipeline
=====================================

This script runs the complete end-to-end machine learning pipeline for stock market prediction.
It consolidates functionality from three main phases:
1. Dataset Creation: Data collection and feature engineering
2. Data Transformation: Preprocessing and target creation
3. Model Training: ML pipeline with hyperparameter tuning

Author: Stamatis Karlos
Course: Stock Market Analytics Zoomcamp 2025
"""

import os
import sys
import warnings
from datetime import datetime
from functools import reduce
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import yaml
import yfinance as yf
from bs4 import BeautifulSoup

# Add utils directory to path
sys.path.append(os.path.abspath('.'))

from utils.data import (
    clean_market_cap,
    features_based_on_fundamentals,
    features_based_on_price,
    get_date_of_previous_month,
    get_economic_indicators_fred,
    get_macro_market_data,
    get_talib_momentum_indicators,
    get_talib_pattern_indicators,
    preprocessing_cyclical_features,
    preprocessing_missing_values,
    temporal_split,
)
from utils.model import (
    data_split,
    get_predictions_correctness,
    tune_and_select_best_classifier,
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Display settings for pandas
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 20)

class StockAnalyticsPipeline:
    """Complete pipeline for stock market analytics and prediction."""

    def __init__(self):
        """Initialize the pipeline with configuration settings."""
        self.config = self._load_config()
        self.data_dir = Path('data')
        self.data_dir.mkdir(exist_ok=True)

        # Constants
        self.STOCKS_URL = "https://stockanalysis.com/list/nasdaq-100-stocks/"
        self.START_DATE = '2020-01-01'
        self.END_DATE = get_date_of_previous_month()

        # Validate dates
        assert pd.to_datetime(self.END_DATE) < pd.to_datetime('today'), "END_DATE must be in the past"
        assert pd.to_datetime(self.START_DATE) < pd.to_datetime(self.END_DATE), "START_DATE must be before END_DATE"

        print(f"🚀 Stock Analytics Pipeline Initialized")
        print(f"📅 Analysis Period: {self.START_DATE} to {self.END_DATE}")
        print(f"📂 Data Directory: {self.data_dir.absolute()}")
        print("-" * 60)

    def _load_config(self):
        """Load configuration from YAML files."""
        config = {}

        # Load external indicators config
        with open('configs/external_indicators.yaml', 'r') as file:
            ext_config = yaml.safe_load(file)
            config['fred_series'] = ext_config['fred_series']
            config['tickers_macro'] = ext_config['tickers_macro']

        # Load preprocessing variables config
        with open('configs/preprocessing_variables.yaml', 'r') as file:
            preprocess_config = yaml.safe_load(file)
            config['dummy_variables'] = preprocess_config['DUMMY_VARIABLES']
            config['drop_variables'] = preprocess_config['DROP_VARIABLES']
            config['cyclical_variables'] = preprocess_config['CYCLICAL_VARIABLES']

        return config

    def fetch_top_stocks(self, top_n=24):
        """
        Fetch top N stocks by market cap from NASDAQ-100.

        Args:
            top_n (int): Number of top stocks to select

        Returns:
            list: List of stock tickers
        """
        print(f"📊 Fetching top {top_n} stocks by market cap...")

        response = requests.get(self.STOCKS_URL)
        soup = BeautifulSoup(response.text, "lxml")

        table = soup.find('table')
        rows = table.find_all('tr')

        # Extract headers and data
        headers = [th.text.strip() for th in rows[0].find_all('th')]
        data = []
        for row in rows[1:]:
            cols = [td.text.strip() for td in row.find_all('td')]
            if cols:
                data.append(cols)

        df = pd.DataFrame(data, columns=headers)

        # Clean and sort by market cap
        df = clean_market_cap(df)
        df.sort_values(by='Market Cap', ascending=False, inplace=True)
        top_stocks = df.head(top_n)

        tickers_list = top_stocks['Symbol'].tolist()

        print(f"✅ Selected {len(tickers_list)} stocks:")
        print(top_stocks[['Symbol', 'Company Name', 'Market Cap']].to_string(index=False))
        print("-" * 60)

        return tickers_list

    def collect_price_data(self, tickers_list):
        """
        Collect and process price data for all tickers.

        Args:
            tickers_list (list): List of stock tickers

        Returns:
            dict: Dictionary containing price data for each ticker
        """
        print("📈 Collecting price data and features...")

        ticker_data = {}
        dataset_price = {}
        dataset_price_df = pd.DataFrame()

        for i, ticker in enumerate(tickers_list):
            print(f"  [{i+1}/{len(tickers_list)}] Processing {ticker}...")

            try:
                # Fetch raw data
                ticker_data[ticker] = yf.Ticker(ticker).history(
                    start=self.START_DATE,
                    end=self.END_DATE,
                    interval='1d'
                )

                # Process price features
                price_data_by_ticker = features_based_on_price(ticker_data[ticker], ticker=ticker)
                dataset_price_df = pd.concat([dataset_price_df, price_data_by_ticker], axis=0)
                dataset_price[ticker] = price_data_by_ticker

            except Exception as e:
                print(f"    ⚠️  Error processing {ticker}: {e}")
                continue

        print(f"✅ Price data collected for {len(dataset_price)} stocks")
        print(f"📊 Total records: {len(dataset_price_df)}")
        print("-" * 60)

        return ticker_data, dataset_price

    def collect_fundamental_data(self, tickers_list):
        """
        Collect fundamental data for all tickers.

        Args:
            tickers_list (list): List of stock tickers

        Returns:
            dict: Dictionary containing fundamental data for each ticker
        """
        print("💼 Collecting fundamental data...")

        dataset_fundamentals = {}

        for i, ticker in enumerate(tickers_list):
            print(f"  [{i+1}/{len(tickers_list)}] Processing {ticker}...")

            try:
                dataset_fundamentals[ticker] = features_based_on_fundamentals(ticker, self.END_DATE)
            except Exception as e:
                print(f"    ⚠️  Error processing {ticker}: {e}")
                continue

        print(f"✅ Fundamental data collected for {len(dataset_fundamentals)} stocks")
        print("-" * 60)

        return dataset_fundamentals

    def collect_technical_indicators(self, ticker_data, tickers_list):
        """
        Calculate technical indicators using TA-Lib.

        Args:
            ticker_data (dict): Raw ticker data
            tickers_list (list): List of stock tickers

        Returns:
            tuple: Pattern and momentum indicators dictionaries
        """
        print("📊 Calculating technical indicators (TA-Lib)...")

        dataset_talib_pattern = {}
        dataset_talib_momentum = {}

        for i, ticker in enumerate(tickers_list):
            if ticker not in ticker_data:
                continue

            print(f"  [{i+1}/{len(tickers_list)}] Processing {ticker}...")

            try:
                # Prepare data for TA-Lib
                data_with_date = ticker_data[ticker].reset_index()
                data_with_date_ticker = data_with_date.copy()
                data_with_date_ticker['Ticker'] = ticker

                # Ensure float64 format
                for col in ['Open', 'High', 'Low', 'Close']:
                    data_with_date_ticker[col] = pd.to_numeric(data_with_date_ticker[col], errors='coerce')

                # Calculate indicators
                dataset_talib_pattern[ticker] = get_talib_pattern_indicators(data_with_date_ticker)
                dataset_talib_momentum[ticker] = get_talib_momentum_indicators(data_with_date_ticker)

            except Exception as e:
                print(f"    ⚠️  Error processing {ticker}: {e}")
                continue

        print(f"✅ Technical indicators calculated for {len(dataset_talib_pattern)} stocks")
        print("-" * 60)

        return dataset_talib_pattern, dataset_talib_momentum

    def collect_external_data(self):
        """
        Collect external macroeconomic and market data.

        Returns:
            tuple: FRED economic data and macro market data
        """
        print("🌍 Collecting external market data...")

        # FRED economic indicators
        print("  📊 Fetching FRED economic indicators...")
        dataset_fred = get_economic_indicators_fred(
            fred_mapping=self.config['fred_series'],
            start_date=self.START_DATE,
            end_date=self.END_DATE
        )

        # Major market indices and ETFs
        print("  📈 Fetching major market indices...")
        dataset_major_indices = get_macro_market_data(
            tickers_macro=self.config['tickers_macro'],
            start_date=self.START_DATE,
            end_date=self.END_DATE
        )

        print(f"✅ External data collected")
        print(f"  📊 FRED indicators: {dataset_fred.shape}")
        print(f"  📈 Market indices: {dataset_major_indices.shape}")
        print("-" * 60)

        return dataset_fred, dataset_major_indices

    def create_full_dataset(self, tickers_list, dataset_price, dataset_fundamentals,
                          dataset_talib_momentum, dataset_talib_pattern,
                          dataset_fred, dataset_major_indices):
        """
        Merge all data sources into a complete dataset.

        Returns:
            pd.DataFrame: Complete merged dataset
        """
        print("🔗 Merging all data sources...")

        full_dataset = pd.DataFrame()

        for i, ticker in enumerate(tickers_list):
            print(f"  [{i+1}/{len(tickers_list)}] Merging data for {ticker}...")

            try:
                if ticker not in dataset_price:
                    continue

                # Start with price data
                ticker_data = dataset_price[ticker].copy()

                # Add fundamentals data
                if ticker in dataset_fundamentals:
                    for k, v in dataset_fundamentals[ticker].items():
                        ticker_data[k] = v

                # Resample technical indicators to monthly frequency
                if ticker in dataset_talib_momentum:
                    df1 = (dataset_talib_momentum[ticker]
                          .set_index('Date')
                          .drop(columns='Ticker')
                          .resample('ME')
                          .median()
                          .reset_index()
                          .dropna())
                else:
                    df1 = pd.DataFrame()

                if ticker in dataset_talib_pattern:
                    df2 = (dataset_talib_pattern[ticker]
                          .set_index('Date')
                          .drop(columns='Ticker')
                          .resample('ME')
                          .median()
                          .reset_index()
                          .dropna())
                else:
                    df2 = pd.DataFrame()

                # Merge all data
                merged = ticker_data

                if not df1.empty:
                    merged = pd.merge(merged, df1, how='left', on='Date', validate='many_to_one')

                if not df2.empty:
                    merged = pd.merge(merged, df2, how='left', on='Date', validate='many_to_one')

                if not dataset_fred.empty:
                    merged = pd.merge(merged, dataset_fred, how='left', on='Date', validate='many_to_one')

                if not dataset_major_indices.empty:
                    merged = pd.merge(merged, dataset_major_indices, how='left', on='Date', validate='many_to_one')

                # Sort by date and create target
                merged = merged.sort_values('Date')
                merged['TARGET'] = merged['Close'].shift(-3)  # 3 months ahead
                merged = merged.dropna(subset=['TARGET'])

                # Append to full dataset
                full_dataset = pd.concat([full_dataset, merged], axis=0)

            except Exception as e:
                print(f"    ⚠️  Error merging {ticker}: {e}")
                continue

        print(f"✅ Full dataset created")
        print(f"  📊 Shape: {full_dataset.shape}")
        print(f"  🎯 Tickers: {full_dataset['ticker'].nunique()}")
        print("-" * 60)

        return full_dataset

    def preprocess_data(self, dataset):
        """
        Apply preprocessing steps to the dataset.

        Args:
            dataset (pd.DataFrame): Raw dataset

        Returns:
            pd.DataFrame: Preprocessed dataset
        """
        print("🔧 Preprocessing data...")

        # Handle missing values
        print("  🔍 Handling missing values...")
        dataset = preprocessing_missing_values(dataset)

        # Apply cyclical transformations
        print("  🔄 Applying cyclical transformations...")
        for variable in self.config['cyclical_variables']:
            dataset = preprocessing_cyclical_features(dataset, variable)

        # One-hot encode categorical variables
        print("  🏷️  One-hot encoding categorical variables...")
        dataset = pd.get_dummies(dataset, columns=self.config['dummy_variables'], drop_first=True)

        # Create target variable for classification
        print("  🎯 Creating binary target variable...")
        dataset['is_positive_growth_3m_future'] = dataset.apply(
            lambda row: 1 if (row['TARGET'] >= row['Close'] * 1.10) else 0,
            axis=1
        )

        # Create temporal splits
        print("  📅 Creating temporal data splits...")
        dataset['Date'] = pd.to_datetime(dataset['Date'])
        min_date_df = dataset.Date.min()
        max_date_df = dataset.Date.max()

        dataset = temporal_split(dataset, min_date=min_date_df, max_date=max_date_df)

        # Drop unnecessary columns
        print("  🗑️  Dropping unnecessary columns...")
        dataset.drop(columns=self.config['drop_variables'], inplace=True)

        split_counts = dataset['split'].value_counts()
        target_dist = dataset['is_positive_growth_3m_future'].value_counts()

        print(f"✅ Preprocessing completed")
        print(f"  📊 Final shape: {dataset.shape}")
        print(f"  📅 Split distribution: {dict(split_counts)}")
        print(f"  🎯 Target distribution: {dict(target_dist)}")
        print("-" * 60)

        return dataset

    def train_model(self, dataset):
        """
        Train and select the best machine learning model.

        Args:
            dataset (pd.DataFrame): Preprocessed dataset

        Returns:
            tuple: Best model, model name, parameters, and metric
        """
        print("🤖 Training machine learning models...")

        # Prepare data for modeling
        print("  📊 Preparing data splits...")
        X_train_valid, y_train_valid, X_test, y_test, scaler = data_split(dataset)

        print(f"  📏 Training set: {X_train_valid.shape}")
        print(f"  📏 Test set: {X_test.shape}")

        # Train and tune models
        print("  🔍 Hyperparameter tuning and model selection...")

        script_dir = os.path.dirname(os.path.abspath(__file__))
        custom_folder = os.path.join(script_dir, 'saved_models')
        print(f"  💾 Models will be saved to: {custom_folder}")

        best_model, best_model_name, best_params, best_metric = tune_and_select_best_classifier(
            X_train_valid, y_train_valid, X_test, y_test, folder_to_save=custom_folder
        )

        # Evaluate final model
        print("  📊 Final model evaluation...")
        y_pred = best_model.predict(X_test)
        get_predictions_correctness(y_test, y_pred, to_predict='is_positive_growth_3m_future')

        print(f"✅ Model training completed")
        print(f"  🏆 Best model: {best_model_name}")
        print(f"  📊 Test precision: {best_metric:.3f}")
        print("-" * 60)

        return best_model, best_model_name, best_params, best_metric

    def save_data(self, full_dataset, processed_dataset):
        """
        Save datasets to CSV files.

        Args:
            full_dataset (pd.DataFrame): Complete raw dataset
            processed_dataset (pd.DataFrame): Preprocessed dataset
        """
        print("💾 Saving datasets...")

        # Save full dataset
        full_path = self.data_dir / 'full_dataset.csv'
        full_dataset.to_csv(full_path, index=False)
        print(f"  📁 Full dataset saved: {full_path}")

        # Save processed dataset
        processed_path = self.data_dir / 'dataset_for_modeling.csv'
        processed_dataset.to_csv(processed_path, index=False)
        print(f"  📁 Processed dataset saved: {processed_path}")

        print("✅ All data saved successfully")
        print("-" * 60)

    def run_pipeline(self, top_n_stocks=24):
        """
        Execute the complete pipeline.

        Args:
            top_n_stocks (int): Number of top stocks to analyze
        """
        print("🚀 Starting Stock Analytics Pipeline")
        print("=" * 60)

        try:
            # Phase 1: Data Collection
            print("PHASE 1: DATA COLLECTION")
            print("=" * 60)

            tickers_list = self.fetch_top_stocks(top_n_stocks)
            ticker_data, dataset_price = self.collect_price_data(tickers_list)
            dataset_fundamentals = self.collect_fundamental_data(tickers_list)
            dataset_talib_pattern, dataset_talib_momentum = self.collect_technical_indicators(ticker_data, tickers_list)
            dataset_fred, dataset_major_indices = self.collect_external_data()

            # Phase 2: Dataset Creation
            print("PHASE 2: DATASET CREATION")
            print("=" * 60)

            full_dataset = self.create_full_dataset(
                tickers_list, dataset_price, dataset_fundamentals,
                dataset_talib_momentum, dataset_talib_pattern,
                dataset_fred, dataset_major_indices
            )

            # Phase 3: Data Preprocessing
            print("PHASE 3: DATA PREPROCESSING")
            print("=" * 60)

            processed_dataset = self.preprocess_data(full_dataset.copy())

            # Phase 4: Model Training
            print("PHASE 4: MODEL TRAINING")
            print("=" * 60)

            best_model, best_model_name, best_params, best_metric = self.train_model(processed_dataset)

            # Phase 5: Save Results
            print("PHASE 5: SAVING RESULTS")
            print("=" * 60)

            self.save_data(full_dataset, processed_dataset)

            # Final Summary
            print("PIPELINE COMPLETED SUCCESSFULLY! 🎉")
            print("=" * 60)
            print(f"📊 Total records processed: {len(full_dataset):,}")
            print(f"🎯 Stocks analyzed: {full_dataset['ticker'].nunique()}")
            print(f"🔢 Features engineered: {processed_dataset.shape[1] - 2}")  # Excluding target and split
            print(f"🏆 Best model: {best_model_name}")
            print(f"📈 Test precision: {best_metric:.1%}")
            print("=" * 60)

            return {
                'full_dataset': full_dataset,
                'processed_dataset': processed_dataset,
                'best_model': best_model,
                'model_name': best_model_name,
                'model_params': best_params,
                'test_precision': best_metric
            }

        except Exception as e:
            print(f"❌ Pipeline failed: {str(e)}")
            raise


def main():
    """Main execution function."""
    print("Stock Market Analytics - Main Pipeline")
    print("====================================")
    print()

    # Initialize and run pipeline
    pipeline = StockAnalyticsPipeline()
    results = pipeline.run_pipeline(top_n_stocks=24)

    print("🎯 Pipeline execution completed!")
    print(f"   Results available in: {pipeline.data_dir.absolute()}")
    print(f"   Model configurations: configs/")


if __name__ == "__main__":
    main()
