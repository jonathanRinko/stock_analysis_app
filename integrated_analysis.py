"""
Integrated Analysis - Combines Fundamental Valuation and Strategy Backtesting
Provides comprehensive stock analysis with both valuation metrics and trading strategy performance
"""
import yfinance as yf
import yaml
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from stock_valuation_app import EnhancedStockValuationApp
from src.backtesting import BacktestEngine, BacktestConfig
from src.strategies import MovingAverageCrossover

def load_backtest_config(config_path='configs/backtest_config.yaml'):
    """Load backtesting configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_integrated_analysis(
    ticker,
    competitors=None,
    run_backtest_flag=True,
    export_valuation=True,
    generate_charts=True
):
    """
    Run comprehensive integrated analysis combining valuation and backtesting

    Args:
        ticker: Stock ticker symbol
        competitors: List of competitor tickers (optional)
        run_backtest_flag: Whether to run backtesting analysis
        export_valuation: Export valuation report to Excel
        generate_charts: Generate charts for both analyses
    """
    print("\n" + "="*80)
    print(f"INTEGRATED ANALYSIS: {ticker}")
    print("Fundamental Valuation + Strategy Backtesting")
    print("="*80 + "\n")

    # ==================== PART 1: FUNDAMENTAL VALUATION ====================
    print("="*80)
    print("PART 1: FUNDAMENTAL VALUATION ANALYSIS")
    print("="*80 + "\n")

    try:
        # Initialize valuation app
        print(f"[OK] Initializing valuation analysis for {ticker}...")
        valuation_app = EnhancedStockValuationApp(ticker)

        # Set competitors
        if competitors:
            valuation_app.get_competitors(competitors)
        else:
            valuation_app.get_competitors()  # Use default competitors

        # Get key metrics
        metrics = valuation_app.get_key_metrics()
        print(f"\n*** KEY METRICS")
        print(f"Current Price: ${metrics.get('Current Price', 'N/A')}")
        print(f"Market Cap: ${metrics.get('Market Cap', 'N/A'):,}" if isinstance(metrics.get('Market Cap'), (int, float)) else f"Market Cap: N/A")
        print(f"PE Ratio: {metrics.get('PE Ratio (TTM)', 'N/A')}")
        print(f"ROE: {metrics.get('ROE', 'N/A')}")

        # DCF Valuation
        dcf = valuation_app.dcf_valuation()
        if dcf:
            print(f"\n*** DCF VALUATION")
            print(f"Intrinsic Value: ${dcf['Intrinsic Value per Share']:.2f}")
            print(f"Current Price: ${dcf['Current Price']:.2f}")
            print(f"Upside/Downside: {dcf['Upside/Downside %']:.2f}%")

            # Investment recommendation based on valuation
            upside = dcf['Upside/Downside %']
            if upside > 20:
                valuation_signal = "UNDERVALUED - Strong Buy"
            elif upside > 10:
                valuation_signal = "UNDERVALUED - Buy"
            elif upside > -10:
                valuation_signal = "FAIR VALUE - Hold"
            elif upside > -20:
                valuation_signal = "OVERVALUED - Consider Selling"
            else:
                valuation_signal = "OVERVALUED - Sell"

            print(f"Valuation Signal: {valuation_signal}")

        # Financial Health
        health = valuation_app.financial_health_score()
        if health:
            print(f"\n*** FINANCIAL HEALTH")
            print(f"Score: {health['Financial Health Score']} ({health['Percentage']})")
            print(f"Rating: {health['Rating']}")

        # Historical Analysis
        hist = valuation_app.historical_trend_analysis()
        if hist:
            print(f"\n*** HISTORICAL PERFORMANCE (5 Years)")
            print(f"Total Return: {hist['summary']['Total Return']:.2f}%")
            print(f"Annualized Return: {hist['summary']['Annualized Return']:.2f}%")
            print(f"Sharpe Ratio: {hist['summary']['Sharpe Ratio']:.2f}")
            print(f"Max Drawdown: {hist['summary']['Max Drawdown']:.2f}%")

        # Export valuation report
        if export_valuation:
            print(f"\n[OK] Exporting valuation report to Excel...")
            valuation_app.export_to_excel()

        # Generate valuation charts
        if generate_charts:
            print(f"[OK] Generating valuation charts...")
            valuation_app.plot_historical_trends()
            valuation_app.plot_technical_indicators()
            valuation_app.plot_monte_carlo()

    except Exception as e:
        print(f"[ERROR] Valuation analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # ==================== PART 2: STRATEGY BACKTESTING ====================
    if run_backtest_flag:
        print(f"\n{'='*80}")
        print("PART 2: STRATEGY BACKTESTING ANALYSIS")
        print("="*80 + "\n")

        try:
            # Load backtest config
            config = load_backtest_config()
            backtest_cfg = config['backtest']
            strategy_cfg = config['strategies']['ma_crossover']

            # Fetch data for backtesting
            print(f"[OK] Fetching historical data for backtesting...")
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(period="2y")  # Use 2 years for backtesting

            # Rename columns to lowercase
            data.columns = [col.lower() for col in data.columns]

            if data.empty:
                print(f"[ERROR] No data available for backtesting")
                return

            # Create strategy
            strategy = MovingAverageCrossover(params=strategy_cfg)
            print(f"[OK] Strategy: MA Crossover (Fast={strategy_cfg['fast_period']}, Slow={strategy_cfg['slow_period']})")

            # Create backtest config
            bt_config = BacktestConfig(
                initial_capital=backtest_cfg['initial_capital'],
                commission=backtest_cfg['commission'],
                slippage=backtest_cfg['slippage'],
                position_size=backtest_cfg['position_size'],
                max_positions=backtest_cfg['max_positions']
            )

            # Run backtest
            print(f"[OK] Running backtest...")
            engine = BacktestEngine(bt_config)
            backtest_result = engine.run(strategy, data)

            # Display backtest results
            bt_metrics = backtest_result.metrics
            print(f"\n*** BACKTEST PERFORMANCE")
            print(f"Total Return: {bt_metrics['total_return']:.2f}%")
            print(f"Annualized Return: {bt_metrics['annualized_return']:.2f}%")
            print(f"Sharpe Ratio: {bt_metrics['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {bt_metrics['max_drawdown']:.2f}%")
            print(f"Win Rate: {bt_metrics['win_rate']:.2f}%")
            print(f"Number of Trades: {len(backtest_result.trades)}")

            # Strategy signal
            if bt_metrics['sharpe_ratio'] > 1.5:
                strategy_signal = "STRONG - Excellent Risk-Adjusted Returns"
            elif bt_metrics['sharpe_ratio'] > 1.0:
                strategy_signal = "GOOD - Solid Risk-Adjusted Returns"
            elif bt_metrics['sharpe_ratio'] > 0.5:
                strategy_signal = "MODERATE - Acceptable Performance"
            else:
                strategy_signal = "WEAK - Poor Risk-Adjusted Returns"

            print(f"Strategy Signal: {strategy_signal}")

        except Exception as e:
            print(f"[ERROR] Backtesting analysis failed: {e}")
            import traceback
            traceback.print_exc()
            backtest_result = None
    else:
        backtest_result = None

    # ==================== PART 3: INTEGRATED RECOMMENDATION ====================
    print(f"\n{'='*80}")
    print("INTEGRATED INVESTMENT RECOMMENDATION")
    print("="*80 + "\n")

    try:
        # Combine signals
        print("*** ANALYSIS SUMMARY")
        print(f"Fundamental Valuation: {valuation_signal if dcf else 'N/A'}")
        print(f"Technical Strategy: {strategy_signal if backtest_result else 'N/A'}")

        # Overall recommendation
        print(f"\n*** OVERALL RECOMMENDATION")

        if dcf and backtest_result:
            # Both analyses available
            if "Buy" in valuation_signal and bt_metrics['sharpe_ratio'] > 1.0:
                overall = "STRONG BUY - Both fundamental and technical signals are positive"
            elif "Buy" in valuation_signal or bt_metrics['sharpe_ratio'] > 1.0:
                overall = "BUY - At least one positive signal"
            elif "Sell" in valuation_signal and bt_metrics['sharpe_ratio'] < 0.5:
                overall = "SELL - Both signals are negative"
            else:
                overall = "HOLD - Mixed signals, wait for better entry"
        elif dcf:
            # Only valuation available
            overall = valuation_signal
        elif backtest_result:
            # Only backtest available
            overall = strategy_signal
        else:
            overall = "INSUFFICIENT DATA"

        print(overall)

        # Risk considerations
        print(f"\n*** RISK CONSIDERATIONS")
        if hist:
            print(f"Historical Volatility: {hist['summary']['Volatility (Annual)']:.2f}%")
        if metrics.get('Beta'):
            beta = metrics['Beta']
            if beta > 1.5:
                risk_level = "HIGH (Very Volatile)"
            elif beta > 1.0:
                risk_level = "ABOVE AVERAGE (More volatile than market)"
            elif beta > 0.5:
                risk_level = "AVERAGE"
            else:
                risk_level = "LOW (Less volatile than market)"
            print(f"Beta: {beta:.2f} - {risk_level}")

        print(f"\n{'='*80}")
        print("DISCLAIMER: This analysis is for educational purposes only.")
        print("Not financial advice. Always do your own research before investing.")
        print("="*80 + "\n")

    except Exception as e:
        print(f"[ERROR] Failed to generate integrated recommendation: {e}")

def main():
    """Main function for interactive integrated analysis"""
    print("\n" + "="*80)
    print(" "*15 + "INTEGRATED STOCK ANALYSIS TOOL")
    print(" "*10 + "Fundamental Valuation + Strategy Backtesting")
    print("="*80 + "\n")

    # Get user input
    ticker = input("Enter stock ticker: ").strip().upper()

    if not ticker:
        print("[ERROR] Please enter a valid ticker symbol.")
        return

    # Ask for competitors
    comp_input = input("Enter competitor tickers (comma-separated, or press Enter to skip): ").strip()
    competitors = [c.strip().upper() for c in comp_input.split(',')] if comp_input else None

    # Ask for options
    print("\nAnalysis Options:")
    run_backtest = input("Run strategy backtesting? (yes/no, default: yes): ").strip().lower() not in ['no', 'n']
    export_excel = input("Export valuation to Excel? (yes/no, default: yes): ").strip().lower() not in ['no', 'n']
    generate_charts = input("Generate charts? (yes/no, default: yes): ").strip().lower() not in ['no', 'n']

    # Run integrated analysis
    run_integrated_analysis(
        ticker=ticker,
        competitors=competitors,
        run_backtest_flag=run_backtest,
        export_valuation=export_excel,
        generate_charts=generate_charts
    )

if __name__ == "__main__":
    main()
