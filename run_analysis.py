from stock_valuation_app import EnhancedStockValuationApp

# Analyze a single stock
ticker = "NVDA"
app = EnhancedStockValuationApp(ticker)

# Set competitors
app.get_competitors(['AMD', 'TSM', 'AVGO'])

# Generate full report with all features
app.generate_full_report(
    export_excel=True,      # Export to Excel
    export_pdf=False,        # Export to PDF
    include_charts=True     # Include charts in reports
)