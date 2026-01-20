from stock_valuation_app import EnhancedStockValuationApp

# Analyze a single stock
ticker = "AAPL"
app = EnhancedStockValuationApp(ticker)

# Set competitors
app.get_competitors(['MSFT', 'GOOGL', 'META'])

# Generate full report with all features
app.generate_full_report(
    export_excel=True,      # Export to Excel
    export_pdf=False,        # Export to PDF
    include_charts=True     # Include charts in reports
)