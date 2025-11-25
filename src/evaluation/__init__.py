"""
Evaluation module for multi-ticker backtesting

Provides:
- TradeLogger: Log individual trades to JSON
- ResultAggregator: Aggregate results across tickers/strategies
- MultiTickerEvaluator: Orchestrate parallel evaluation
- Enhanced metrics: Risk-adjusted performance, regime analysis
- Enhanced visualizations: Comprehensive charts
- Report generator: Markdown evaluation reports
"""

from src.evaluation.trade_logger import TradeLogger
from src.evaluation.result_aggregator import ResultAggregator
from src.evaluation.multi_ticker_evaluator import (
    MultiTickerEvaluator,
    EvaluationConfig,
    run_quick_test,
    run_full_evaluation,
)
from src.evaluation.enhanced_metrics import (
    MarketRegime,
    RiskAdjustedMetrics,
    DecisionQualityMetrics,
    RegimePerformance,
    detect_market_regime,
    calculate_risk_adjusted_metrics,
    calculate_decision_quality_metrics,
    calculate_regime_stratified_performance,
    generate_metrics_summary,
)
from src.evaluation.enhanced_visualizations import create_enhanced_visualizations
from src.evaluation.report_generator import (
    EvaluationReportGenerator,
    generate_evaluation_report,
)

__all__ = [
    # Core components
    "TradeLogger",
    "ResultAggregator",
    "MultiTickerEvaluator",
    "EvaluationConfig",
    "run_quick_test",
    "run_full_evaluation",
    # Enhanced metrics
    "MarketRegime",
    "RiskAdjustedMetrics",
    "DecisionQualityMetrics",
    "RegimePerformance",
    "detect_market_regime",
    "calculate_risk_adjusted_metrics",
    "calculate_decision_quality_metrics",
    "calculate_regime_stratified_performance",
    "generate_metrics_summary",
    # Visualizations
    "create_enhanced_visualizations",
    # Report generation
    "EvaluationReportGenerator",
    "generate_evaluation_report",
]
