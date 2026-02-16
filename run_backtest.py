"""
Backtesting Script - Run strategy backtests
Loads configuration from configs/backtest_config.yaml
"""
import yfinance as yf
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

from src.backtesting import BacktestEngine, BacktestConfig
from src.strategies import MovingAverageCrossover

def load_config(config_path='configs/backtest_config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def fetch_data(symbol, start_date, end_date):
    """Fetch historical data for backtesting"""
    print(f"[OK] Fetching data for {symbol} from {start_date} to {end_date}...")

    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date)

    # Rename columns to lowercase for consistency
    data.columns = [col.lower() for col in data.columns]

    if data.empty:
        raise ValueError(f"No data available for {symbol}")

    print(f"[OK] Fetched {len(data)} bars")
    return data

def plot_backtest_results(result, symbol, output_dir='backtest_results'):
    """Visualize backtest results"""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle(f'{symbol} Backtest Results', fontsize=16, fontweight='bold')

    # 1. Equity Curve
    ax1 = axes[0]
    equity = result.equity_curve
    ax1.plot(equity['timestamp'], equity['equity'], linewidth=2, color='blue', label='Portfolio Value')
    ax1.axhline(y=result.metrics['starting_equity'], color='gray', linestyle='--', alpha=0.5, label='Starting Capital')
    ax1.set_title('Equity Curve')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Drawdown
    ax2 = axes[1]
    equity_values = equity['equity'].values
    peak = pd.Series(equity_values).expanding().max()
    drawdown = (equity_values - peak) / peak * 100
    ax2.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
    ax2.plot(drawdown, color='red', linewidth=1)
    ax2.set_title('Drawdown')
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True, alpha=0.3)

    # 3. Returns Distribution
    ax3 = axes[2]
    if not result.trades.empty:
        returns = result.trades['return_pct'] * 100
        ax3.hist(returns, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax3.axvline(x=returns.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {returns.mean():.2f}%')
        ax3.set_title('Trade Returns Distribution')
        ax3.set_xlabel('Return (%)')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    filename = os.path.join(output_dir, f'{symbol}_backtest.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"[OK] Chart saved to {filename}")

    plt.close()

def print_backtest_report(result, symbol, strategy_name):
    """Print comprehensive backtest report"""
    print("\n" + "="*80)
    print(f"BACKTEST REPORT: {symbol} - {strategy_name}")
    print("="*80)

    metrics = result.metrics

    print("\n*** PERFORMANCE METRICS")
    print("-" * 80)
    print(f"Total Return.................... {metrics['total_return']:.2f}%")
    print(f"Annualized Return............... {metrics['annualized_return']:.2f}%")
    print(f"Average Return.................. {metrics['avg_return']:.2f}%")
    print(f"Final Equity.................... ${metrics['final_equity']:,.2f}")
    print(f"Starting Equity................. ${metrics['starting_equity']:,.2f}")

    print("\n*** RISK METRICS")
    print("-" * 80)
    print(f"Volatility (Annual)............. {metrics['volatility']:.2f}%")
    print(f"Maximum Drawdown................ {metrics['max_drawdown']:.2f}%")
    print(f"Sharpe Ratio.................... {metrics['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio................... {metrics['sortino_ratio']:.2f}")
    print(f"Calmar Ratio.................... {metrics['calmar_ratio']:.2f}")

    print("\n*** TRADE STATISTICS")
    print("-" * 80)
    print(f"Number of Trades................ {len(result.trades)}")
    print(f"Win Rate........................ {metrics['win_rate']:.2f}%")
    print(f"Average Win..................... {metrics['avg_win']:.2f}%")
    print(f"Average Loss.................... {metrics['avg_loss']:.2f}%")
    print(f"Profit Factor................... {metrics['profit_factor']:.2f}")
    print(f"Best Day........................ {metrics['best_day']:.2f}%")
    print(f"Worst Day....................... {metrics['worst_day']:.2f}%")

    # Trade details
    if not result.trades.empty:
        print("\n*** TRADE DETAILS (Last 10)")
        print("-" * 80)
        trade_df = result.trades[['entry_time', 'exit_time', 'entry_price', 'exit_price', 'pnl', 'return_pct']].tail(10)
        trade_df['pnl'] = trade_df['pnl'].apply(lambda x: f"${x:.2f}")
        trade_df['return_pct'] = trade_df['return_pct'].apply(lambda x: f"{x*100:.2f}%")
        print(trade_df.to_string(index=False))

    print("\n" + "="*80)

def save_results(result, symbol, output_dir='backtest_results'):
    """Save backtest results to files"""
    os.makedirs(output_dir, exist_ok=True)

    # Save trades to CSV
    if not result.trades.empty:
        trades_file = os.path.join(output_dir, f'{symbol}_trades.csv')
        result.trades.to_csv(trades_file, index=False)
        print(f"[OK] Trades saved to {trades_file}")

    # Save equity curve to CSV
    equity_file = os.path.join(output_dir, f'{symbol}_equity_curve.csv')
    result.equity_curve.to_csv(equity_file, index=False)
    print(f"[OK] Equity curve saved to {equity_file}")

    # Save metrics to text file
    metrics_file = os.path.join(output_dir, f'{symbol}_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"Backtest Metrics for {symbol}\n")
        f.write("="*60 + "\n\n")
        for key, value in result.metrics.items():
            f.write(f"{key}: {value}\n")
    print(f"[OK] Metrics saved to {metrics_file}")

def run_backtest(symbol, config):
    """Run a single backtest"""
    # Extract configuration
    backtest_cfg = config['backtest']
    data_cfg = config['data']
    strategy_cfg = config['strategies']['ma_crossover']
    output_cfg = config['output']

    # Fetch data
    data = fetch_data(
        symbol=symbol,
        start_date=data_cfg['start_date'],
        end_date=data_cfg['end_date']
    )

    # Create strategy
    strategy = MovingAverageCrossover(params=strategy_cfg)
    print(f"[OK] Strategy initialized: MA Crossover (Fast={strategy_cfg['fast_period']}, Slow={strategy_cfg['slow_period']})")

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
    result = engine.run(strategy, data)

    # Print report
    print_backtest_report(result, symbol, "MA Crossover")

    # Save results if configured
    if output_cfg.get('save_results', False):
        output_dir = output_cfg.get('output_dir', 'backtest_results')

        if output_cfg.get('export_trades', False):
            save_results(result, symbol, output_dir)

        if output_cfg.get('generate_charts', False):
            plot_backtest_results(result, symbol, output_dir)

    return result

def main():
    """Main function"""
    print("\n" + "="*80)
    print(" "*20 + "STOCK STRATEGY BACKTESTER")
    print(" "*25 + "Powered by Python")
    print("="*80 + "\n")

    try:
        # Load configuration
        print("[OK] Loading configuration from configs/backtest_config.yaml...")
        config = load_config()

        # Get symbols to test
        symbols = config['data']['symbols']
        print(f"[OK] Symbols to backtest: {', '.join(symbols)}")

        # Run backtests for each symbol
        results = {}
        for symbol in symbols:
            print(f"\n{'='*80}")
            print(f"BACKTESTING: {symbol}")
            print(f"{'='*80}")

            try:
                result = run_backtest(symbol, config)
                results[symbol] = result
            except Exception as e:
                print(f"[ERROR] Failed to backtest {symbol}: {e}")
                import traceback
                traceback.print_exc()

        # Summary
        if results:
            print(f"\n{'='*80}")
            print("SUMMARY - ALL BACKTESTS")
            print(f"{'='*80}\n")

            summary_data = []
            for symbol, result in results.items():
                summary_data.append({
                    'Symbol': symbol,
                    'Total Return (%)': f"{result.metrics['total_return']:.2f}",
                    'Sharpe Ratio': f"{result.metrics['sharpe_ratio']:.2f}",
                    'Max Drawdown (%)': f"{result.metrics['max_drawdown']:.2f}",
                    'Win Rate (%)': f"{result.metrics['win_rate']:.2f}",
                    'Num Trades': len(result.trades)
                })

            summary_df = pd.DataFrame(summary_data)
            print(summary_df.to_string(index=False))

        print(f"\n{'='*80}")
        print("Backtesting complete!")
        print(f"{'='*80}\n")

    except Exception as e:
        print(f"[ERROR] Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
