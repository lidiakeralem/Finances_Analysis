try:
    from scripts.finance_function import perform_sentiment_analysis, perform_topic_modeling
    print("Imports successful!")
except ImportError as e:
    print(f"Import failed: {e}")