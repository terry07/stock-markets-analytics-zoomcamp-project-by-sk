#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Start Example for Stock Analytics Pipeline
==============================================

This script demonstrates how to use the main pipeline with different configurations.
"""

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import StockAnalyticsPipeline


def run_quick_demo():
    """Run a quick demo with a smaller number of stocks."""
    print("🚀 Running Quick Demo (Top 5 Stocks)")
    print("=" * 50)

    pipeline = StockAnalyticsPipeline()

    # Run with just top 5 stocks for faster execution
    results = pipeline.run_pipeline(top_n_stocks=5)

    print("\n📊 Demo Results Summary:")
    print(f"   Records processed: {len(results['full_dataset']):,}")
    print(f"   Features created: {results['processed_dataset'].shape[1] - 2}")
    print(f"   Best model: {results['model_name']}")
    print(f"   Test precision: {results['test_precision']:.1%}")

def run_full_pipeline():
    """Run the complete pipeline with all 24 stocks."""
    print("🚀 Running Full Pipeline (Top 24 Stocks)")
    print("=" * 50)

    pipeline = StockAnalyticsPipeline()
    results = pipeline.run_pipeline(top_n_stocks=24)

    return results

def analyze_existing_data():
    """Analyze data that's already been created."""
    data_dir = Path('data')

    if (data_dir / 'full_dataset.csv').exists():
        print("📊 Analyzing existing full dataset...")
        df = pd.read_csv(data_dir / 'full_dataset.csv')

        print(f"   Shape: {df.shape}")
        print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"   Tickers: {df['ticker'].nunique()}")
        print(f"   Missing values: {df.isnull().sum().sum()}")

        return df
    else:
        print("❌ No existing dataset found. Run the pipeline first.")
        return None

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

        if mode == 'demo':
            run_quick_demo()
        elif mode == 'full':
            run_full_pipeline()
        elif mode == 'analyze':
            analyze_existing_data()
        else:
            print("Usage: python run_example.py [demo|full|analyze]")
            print("  demo    - Run with top 5 stocks (faster)")
            print("  full    - Run with top 24 stocks (complete)")
            print("  analyze - Analyze existing data")
    else:
        print("Stock Analytics Pipeline - Usage Examples")
        print("=" * 45)
        print("\nAvailable modes:")
        print("  python run_example.py demo    - Quick demo (5 stocks)")
        print("  python run_example.py full    - Full pipeline (24 stocks)")
        print("  python run_example.py analyze - Analyze existing data")
        print("\nOr run the main pipeline directly:")
        print("  python main.py")
