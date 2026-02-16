# Stock Analysis App
**Comprehensive Stock Analysis Tool with Fundamental Valuation & Strategy Backtesting**
By Jonathan Rinko

## Features

### Fundamental Analysis
- **DCF & Multiple Valuation Models** - Intrinsic value calculation using various methods
- **Competitor Comparison** - Side-by-side analysis with industry peers
- **Historical Trend Analysis** - 5-year performance metrics and trends
- **Technical Indicators** - RSI, MACD, Bollinger Bands, Stochastic Oscillator
- **Monte Carlo Simulation** - Price prediction with confidence intervals
- **Financial Health Scoring** - Comprehensive 100-point health assessment
- **Export Capabilities** - Excel and PDF reports with charts

### Strategy Backtesting (NEW!)
- **Trading Strategy Framework** - Modular strategy development system
- **Realistic Simulation** - Commission, slippage, and position sizing
- **Performance Metrics** - Sharpe, Sortino, Calmar ratios, drawdown analysis
- **Walk-Forward Optimization** - Robust parameter optimization
- **Visual Analytics** - Equity curves, drawdown charts, return distributions

## Installation

### Install Requirements
```bash
python -m pip install -r requirements.txt
```

Or install directly to your interpreter using the path:
```bash
where python
/path/to/your/python -m pip install -r requirements.txt
```

## Usage

### 1. Fundamental Valuation Analysis

Update ticker in [run_analysis.py](run_analysis.py):
```python
ticker = "AAPL"
app = EnhancedStockValuationApp(ticker)
app.get_competitors(['MSFT', 'GOOGL'])
app.generate_full_report(export_excel=True, include_charts=True)
```

Run the analysis:
```bash
python run_analysis.py
```

### 2. Strategy Backtesting

Configure your backtest in [configs/backtest_config.yaml](configs/backtest_config.yaml):
```yaml
backtest:
  initial_capital: 100000.0
  commission: 0.001

strategies:
  ma_crossover:
    fast_period: 10
    slow_period: 30

data:
  symbols:
    - AAPL
    - MSFT
  start_date: "2020-01-01"
  end_date: "2023-12-31"
```

Run the backtest:
```bash
python run_backtest.py
```

### 3. Integrated Analysis (Recommended!)

Combines fundamental valuation with strategy backtesting:
```bash
python integrated_analysis.py
```

This will:
1. Perform fundamental valuation analysis
2. Run strategy backtesting
3. Generate combined investment recommendation
4. Export comprehensive reports

### 4. Batch Analysis

Analyze multiple stocks at once using [batch_analysis.py](batch_analysis.py):
```python
tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
```

```bash
python batch_analysis.py
```

### 5. Interactive Mode

For interactive analysis with prompts:
```bash
python stock_valuation_app.py
```

## Project Structure

```
stock_analysis_app/
├── stock_valuation_app.py      # Core valuation engine
├── run_analysis.py              # Single stock analysis
├── batch_analysis.py            # Multiple stock analysis
├── run_backtest.py              # Strategy backtesting
├── integrated_analysis.py       # Combined analysis (NEW!)
├── requirements.txt             # Python dependencies
├── configs/
│   └── backtest_config.yaml    # Backtesting configuration
├── src/
│   ├── strategies/             # Trading strategies
│   │   ├── base.py            # Base strategy class
│   │   └── ma_crossover.py    # MA crossover strategy
│   └── backtesting/           # Backtesting engine
│       ├── engine.py          # Core backtest engine
│       ├── broker.py          # Order execution simulation
│       ├── portfolio.py       # Portfolio management
│       ├── performance.py     # Performance metrics
│       └── walk_forward.py    # Walk-forward optimization
├── charts/                     # Generated charts (git-ignored)
└── backtest_results/          # Backtest outputs (git-ignored)
```

## Output Files

The app generates various output files:

- **Excel Reports** - `{TICKER}_valuation_{DATE}.xlsx`
- **PDF Reports** - `{TICKER}_valuation_{DATE}.pdf`
- **Charts** - PNG images in `charts/` directory
- **Backtest Results** - CSV and charts in `backtest_results/` directory

## Example Output

### Valuation Analysis
```
*** KEY FINANCIAL METRICS
Company Name: Apple Inc.
Current Price: $255.78
PE Ratio: 32.38
ROE: 147.25%

*** DCF VALUATION
Intrinsic Value per Share: $95.69
Current Price: $255.78
Upside/Downside: -79.07%
```

### Backtest Results
```
*** BACKTEST PERFORMANCE
Total Return: 45.32%
Annualized Return: 22.15%
Sharpe Ratio: 1.85
Max Drawdown: -12.34%
Win Rate: 62.50%
```

## Customization

### Creating Custom Strategies

Extend the `BaseStrategy` class in `src/strategies/`:

```python
from src.strategies import BaseStrategy, Signal

class MyStrategy(BaseStrategy):
    def calculate_indicators(self, data):
        # Add your indicators
        data['my_indicator'] = ...
        return data

    def generate_signals(self, data):
        # Generate buy/sell signals
        signals = []
        # ... your logic
        return signals
```

### Modifying Backtest Parameters

Edit [configs/backtest_config.yaml](configs/backtest_config.yaml) to adjust:
- Initial capital
- Commission and slippage rates
- Position sizing
- Date ranges
- Strategy parameters

## Requirements

- Python 3.8+
- yfinance>=0.2.28
- pandas>=1.5.0
- numpy>=1.23.0
- matplotlib>=3.6.0
- seaborn>=0.12.0
- scipy>=1.9.0
- openpyxl>=3.0.0
- reportlab>=3.6.0
- pyyaml>=6.0

## Contributing

Contributions welcome! Areas for enhancement:
- Additional trading strategies
- More valuation models
- Portfolio optimization
- Risk management features
- Real-time data integration

## Disclaimer

**This tool is for educational and research purposes only.** It does not constitute financial advice or investment recommendations. Always conduct your own research and consult with a qualified financial advisor before making investment decisions. Past performance does not guarantee future results.

## <span style="color:#2ea44f">If you like this repo, please give me a star ⭐</span>

