#!/bin/bash
# Quant13 Evaluation Script
# Run from project root: bash docs/evaluation_report/run_evaluation.sh

# Activate virtual environment if exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# 90-Day 10-Ticker Evaluation (Used for Final Report)
python3 -c "
from datetime import datetime
from src.evaluation import MultiTickerEvaluator, EvaluationConfig

config = EvaluationConfig(
    tickers=['SPY', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AMD', 'JPM', 'QQQ'],
    strategies=['quant13', 'technical', 'chimpanzee'],
    start_date=datetime(2024, 9, 1),
    end_date=datetime(2024, 12, 1),
    initial_capital=10000.0,
    position_size_pct=0.90,
    signal_frequency='weekly',
    monitor_frequency='daily',
    parallel_workers=10,
)

print('=' * 80)
print('90-DAY 10-TICKER EVALUATION')
print('=' * 80)
print(f'Period: {config.start_date.date()} to {config.end_date.date()} (90 days)')
print(f'Tickers: {config.tickers}')
print(f'Strategies: {config.strategies}')
print(f'Initial Capital: \${config.initial_capital:,.0f}')
print(f'Parallel Workers: {config.parallel_workers}')
print('=' * 80)
print()

evaluator = MultiTickerEvaluator(config)
results = evaluator.run_evaluation()

print()
print('=' * 80)
print('EVALUATION COMPLETE')
print('=' * 80)
print(f'Results: {results[\"num_results\"]} backtests')
print(f'Output: {results[\"output_dir\"]}')
print('=' * 80)

if 'summaries' in results:
    print()
    print('STRATEGY COMPARISON:')
    print(results['summaries']['strategy_comparison'].to_string())
    print()
    print('PERFORMANCE MATRIX:')
    print(results['summaries']['performance_matrix'].to_string())
"
