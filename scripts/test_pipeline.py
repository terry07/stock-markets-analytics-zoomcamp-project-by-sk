#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for the main pipeline.
This script validates that the main.py can be imported and initialized correctly.
"""

import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_pipeline_import():
    """Test if the pipeline can be imported."""
    try:
        from main import StockAnalyticsPipeline
        print("✅ Successfully imported StockAnalyticsPipeline")
        return True
    except ImportError as e:
        print(f"❌ Failed to import StockAnalyticsPipeline: {e}")
        return False

def test_pipeline_initialization():
    """Test if the pipeline can be initialized."""
    try:
        from main import StockAnalyticsPipeline

        # Check if config files exist
        config_files = [
            'configs/external_indicators.yaml',
            'configs/preprocessing_variables.yaml'
        ]

        for config_file in config_files:
            if not Path(config_file).exists():
                print(f"❌ Missing config file: {config_file}")
                return False

        pipeline = StockAnalyticsPipeline()
        print("✅ Successfully initialized StockAnalyticsPipeline")
        print(f"   📅 Date range: {pipeline.START_DATE} to {pipeline.END_DATE}")
        print(f"   📂 Data directory: {pipeline.data_dir}")
        return True

    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        return False

def test_config_loading():
    """Test configuration loading."""
    try:
        from main import StockAnalyticsPipeline
        pipeline = StockAnalyticsPipeline()

        # Check if config has required keys
        required_keys = ['fred_series', 'tickers_macro', 'dummy_variables', 'drop_variables', 'cyclical_variables']
        for key in required_keys:
            if key not in pipeline.config:
                print(f"❌ Missing config key: {key}")
                return False

        print("✅ Configuration loaded successfully")
        print(f"   📊 FRED series: {len(pipeline.config['fred_series'])}")
        print(f"   📈 Macro tickers: {len(pipeline.config['tickers_macro'])}")
        return True

    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing Stock Analytics Pipeline")
    print("=" * 40)

    tests = [
        ("Import Test", test_pipeline_import),
        ("Initialization Test", test_pipeline_initialization),
        ("Configuration Test", test_config_loading),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"   ⚠️  {test_name} failed")

    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! The pipeline is ready to run.")
        print("\nTo run the full pipeline, execute:")
        print("python main.py")
    else:
        print("❌ Some tests failed. Please check the setup.")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
