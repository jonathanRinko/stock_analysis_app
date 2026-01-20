from stock_valuation_app import EnhancedStockValuationApp

tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]

for ticker in tickers:
    print(f"\n{'='*50}")
    print(f"Analyzing {ticker}")
    print('='*50)
    
    app = EnhancedStockValuationApp(ticker)
    app.generate_full_report(export_excel=True)