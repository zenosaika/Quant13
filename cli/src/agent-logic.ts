import fs from 'fs';
import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import type { AddLogFn, UpdateLogFn } from './types.js';

interface BridgeTradeLeg {
  contract_symbol: string;
  type: string;
  action: string;
  strike_price: number;
  expiration_date: string;
  quantity: number;
}

interface BridgeVolatility {
  iv_rank?: number;
  volatility_forecast?: string;
  skew_analysis?: string;
  term_structure?: string;
}

interface BridgeSentimentArticle {
  title?: string;
  publisher?: string | null;
  link?: string | null;
  sentiment_score?: number;
  rationale?: string | null;
}

interface BridgeSentiment {
  overall_sentiment_score?: number;
  overall_summary?: string;
  articles?: BridgeSentimentArticle[];
}

interface BridgeTechnical {
  llm_report?: Record<string, unknown>;
  derived_signals?: unknown;
  key_levels?: Record<string, unknown>;
  recent_patterns?: unknown;
}

interface BridgeFundamental {
  llm_synthesis?: Record<string, unknown>;
  financial_ratios?: Record<string, unknown>;
  financial_trends?: Array<Record<string, unknown>>;
  qualitative_summary?: Record<string, unknown>;
  business_overview?: Record<string, unknown>;
}

interface BridgeResponse {
  ticker: string;
  generated_at: string;
  trade: {
    strategy_name: string;
    action: string;
    quantity: number;
    trade_legs: BridgeTradeLeg[];
    notes?: string | null;
  };
  thesis: {
    summary: string;
    conviction_level: string;
    key_evidence?: string[];
  };
  risk: {
    final_recommendation: string;
    adjustments: { profile: string; recommendation: string }[];
  };
  results_path?: string | null;
  agents?: {
    volatility?: BridgeVolatility;
    sentiment?: BridgeSentiment;
    technical?: BridgeTechnical;
    fundamental?: BridgeFundamental;
  };
}

const progressMessages = [
  'Deploying satellite feeds across Project 13 grid…',
  'Spinning up bullish vs. bearish research duel…',
  'Composing multi-leg execution schematic…',
  'Calibrating capital shields with risk sentries…',
];

const progressMessageAt = (index: number): string => {
  if (progressMessages.length === 0) {
    return 'Working…';
  }
  const normalized = ((index % progressMessages.length) + progressMessages.length) % progressMessages.length;
  const message = progressMessages[normalized];
  return typeof message === 'string' ? message : 'Working…';
};

const currentFileUrl = new URL('.', import.meta.url);
const currentDir = fileURLToPath(currentFileUrl);
const repoRoot = path.resolve(currentDir, '..', '..');

export async function runAgentLogic(
  userInput: string,
  addLog: AddLogFn,
  updateLog: UpdateLogFn
): Promise<void> {
  const ticker = userInput.trim().toUpperCase();

  addLog({
    agent: 'system',
    headline: 'Project 13 uplink initialized',
    content: `Calibrating analyzers for ${ticker}…`,
    status: 'completed',
  });

  const analysisLogId = addLog({
    agent: 'orchestrator',
    content: progressMessageAt(0),
    status: 'pending',
  });

  let phase = 0;
  const tickerInterval = setInterval(() => {
    phase += 1;
    updateLog(analysisLogId, { content: progressMessageAt(phase) });
  }, 1200);

  let result: BridgeResponse;
  try {
    result = await runPythonBridge(ticker);
  } catch (error) {
    clearInterval(tickerInterval);
    updateLog(analysisLogId, {
      status: 'error',
      content: error instanceof Error ? error.message : 'Pipeline execution failed.',
    });
    throw error;
  }

  clearInterval(tickerInterval);

  updateLog(analysisLogId, {
    status: 'completed',
    headline: `Synthesis locked for ${ticker}`,
    content: 'Agent collective reached consensus.',
  });

  emitVolatilityLog(addLog, result);
  emitSentimentLog(addLog, result);
  emitTechnicalLog(addLog, result);
  emitFundamentalLog(addLog, result);
  emitThesisLog(addLog, result);
  emitTradeLog(addLog, result);
  emitRiskLog(addLog, result);
  emitArtifactsLog(addLog, result);

  addLog({
    agent: 'system',
    headline: 'Cycle complete',
    content: `Project 13 debrief sealed for ${ticker}.`,
    status: 'completed',
  });
}

async function runPythonBridge(ticker: string): Promise<BridgeResponse> {
  const pythonExecutable = resolvePythonExecutable();

  const python = spawn(pythonExecutable, ['-m', 'src.cli_bridge', ticker], {
    cwd: repoRoot,
    stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, PYTHONUNBUFFERED: '1' },
  });

  let stdout = '';
  let stderr = '';

  python.stdout.on('data', (chunk) => {
    stdout += chunk.toString();
  });

  python.stderr.on('data', (chunk) => {
    stderr += chunk.toString();
  });

  const exitCode: number = await new Promise((resolve, reject) => {
    python.on('error', (error) => reject(error));
    python.on('close', (code) => resolve(code ?? 1));
  });

  if (exitCode !== 0) {
    throw new Error(stderr.trim() || `Python pipeline exited with code ${exitCode}`);
  }

  try {
    const parsed = JSON.parse(stdout) as BridgeResponse;
    return parsed;
  } catch (error) {
    throw new Error(
      error instanceof Error
        ? `Failed to parse pipeline output: ${error.message}`
        : 'Failed to parse pipeline output.'
    );
  }
}

function resolvePythonExecutable(): string {
  const overrides = process.env.QUANT13_PYTHON;
  if (overrides && overrides.trim()) {
    return overrides.trim();
  }

  const candidates = [
    path.join(repoRoot, '.venv', 'bin', 'python'),
    path.join(repoRoot, '.venv', 'bin', 'python3'),
    path.join(repoRoot, '.venv', 'Scripts', 'python.exe'),
    path.join(repoRoot, 'venv', 'bin', 'python'),
    path.join(repoRoot, 'venv', 'bin', 'python3'),
    path.join(repoRoot, 'venv', 'Scripts', 'python.exe'),
  ];

  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }

  return process.platform === 'win32' ? 'python' : 'python3';
}

function emitVolatilityLog(addLog: AddLogFn, result: BridgeResponse): void {
  const volatility = result.agents?.volatility;
  if (!volatility) {
    return;
  }

  const metadata: Record<string, unknown> = {
    ivRank: toNumber(volatility.iv_rank),
    forecast: toString(volatility.volatility_forecast),
    skew: toString(volatility.skew_analysis),
    term: toString(volatility.term_structure),
  };

  const content =
    toString(volatility.volatility_forecast) ?? 'Volatility forecast unavailable.';

  addLog({
    agent: 'VolatilityModelingAgent',
    headline: `Turbulence map // ${result.ticker}`,
    content,
    metadata,
    status: 'completed',
  });
}

function emitSentimentLog(addLog: AddLogFn, result: BridgeResponse): void {
  const sentiment = result.agents?.sentiment;
  if (!sentiment) {
    return;
  }

  const summary =
    toString(sentiment.overall_summary) ?? 'No dominant narrative detected.';

  const articles = Array.isArray(sentiment.articles) ? sentiment.articles : [];
  const spotlight = articles.find((article) => toString(article.title));

  const metadata: Record<string, unknown> = {
    summary,
  };
  const score = toNumber(sentiment.overall_sentiment_score);
  if (score !== undefined) {
    metadata.score = score;
  }
  if (spotlight) {
    metadata.spotlight = {
      title: toString(spotlight.title),
      publisher: toString(spotlight.publisher),
    };
  }

  addLog({
    agent: 'SentimentAgent',
    headline: `Narrative pulse // ${result.ticker}`,
    content: summary,
    metadata,
    status: 'completed',
  });
}

function emitTechnicalLog(addLog: AddLogFn, result: BridgeResponse): void {
  const technical = result.agents?.technical;
  if (!technical) {
    return;
  }

  const llmReport = (technical.llm_report ?? {}) as Record<string, unknown>;
  const bias =
    toString(llmReport.technical_bias) ?? toString(llmReport.bias) ?? undefined;
  const summary =
    toString(llmReport.summary) ||
    toString(llmReport.overview) ||
    toString(llmReport.commentary) ||
    'Technical signals evaluated.';

  const highlights = dedupe([
    ...collectStrings(technical.derived_signals),
    ...collectStrings(llmReport.highlights),
    ...collectStrings(llmReport.signals),
    ...collectStrings(technical.recent_patterns),
  ]).slice(0, 4);

  const metadata: Record<string, unknown> = {
    summary,
    highlights,
  };
  if (bias) {
    metadata.bias = bias;
  }

  addLog({
    agent: 'TechnicalAnalyst',
    headline: `Signal lattice // ${result.ticker}`,
    content: summary,
    metadata,
    status: 'completed',
  });
}

function emitFundamentalLog(addLog: AddLogFn, result: BridgeResponse): void {
  const fundamental = result.agents?.fundamental;
  if (!fundamental) {
    return;
  }

  const synthesis = (fundamental.llm_synthesis ?? {}) as Record<string, unknown>;
  const summary =
    toString(synthesis.summary) ||
    toString(synthesis.overall_thesis) ||
    toString(synthesis.investment_case) ||
    'Fundamental profile updated.';

  const ratiosRaw = (fundamental.financial_ratios ?? {}) as Record<string, unknown>;
  const ratioMap: Record<string, number | null> = {};
  ratioMap['P/E'] = toNumber(ratiosRaw.pe_ratio) ?? null;
  ratioMap['P/S'] = toNumber(ratiosRaw.ps_ratio) ?? null;
  ratioMap['D/E'] = toNumber(ratiosRaw.debt_to_equity) ?? null;
  ratioMap['Current'] = toNumber(ratiosRaw.current_ratio) ?? null;

  const highlights = buildFundamentalHighlights(fundamental);

  const metadata: Record<string, unknown> = {
    summary,
    ratios: ratioMap,
    highlights,
  };

  addLog({
    agent: 'FundamentalAnalyst',
    headline: `Balance sheet oracle // ${result.ticker}`,
    content: summary,
    metadata,
    status: 'completed',
  });
}

function emitThesisLog(addLog: AddLogFn, result: BridgeResponse): void {
  const { thesis } = result;
  const evidenceLines = Array.isArray(thesis.key_evidence)
    ? thesis.key_evidence
        .map((line) => toString(line))
        .filter((line): line is string => Boolean(line))
    : [];

  const contentLines = [
    `Conviction ▸ ${thesis.conviction_level}`,
    ...evidenceLines.map((line) => `• ${line}`),
  ];

  addLog({
    agent: 'DebateModerator',
    headline: `Thesis arbiter verdict // ${thesis.summary}`,
    content: contentLines.join('\n'),
    status: 'completed',
  });
}

function emitTradeLog(addLog: AddLogFn, result: BridgeResponse): void {
  const { trade } = result;
  const legDescriptions = trade.trade_legs.map((leg) => {
    const components = [
      `${leg.action} ${leg.quantity} × ${leg.type}`,
      `@ ${leg.strike_price}`,
      `exp ${leg.expiration_date}`,
      leg.contract_symbol ? `(${leg.contract_symbol})` : undefined,
    ];
    return components.filter(Boolean).join(' ');
  });

  const contentLines = [
    `${trade.action} x${trade.quantity}`,
    ...legDescriptions.map((line) => `• ${line}`),
  ];
  if (trade.notes) {
    contentLines.push(`Notes ▸ ${trade.notes}`);
  }

  addLog({
    agent: 'TraderAgent',
    headline: `Execution schematic // ${trade.strategy_name}`,
    content: contentLines.join('\n'),
    status: 'completed',
    metadata: {
      legs: trade.trade_legs,
    },
  });
}

function emitRiskLog(addLog: AddLogFn, result: BridgeResponse): void {
  const { risk, thesis } = result;
  const adjustments = risk.adjustments.map(
    (adj) => `• ${adj.profile}: ${adj.recommendation}`
  );

  const contentLines = [
    `Conviction context ▸ ${thesis.conviction_level}`,
    `Final stance ▸ ${risk.final_recommendation}`,
    ...adjustments,
  ];

  addLog({
    agent: 'RiskManagementTeam',
    headline: 'Risk sentinel overlays',
    content: contentLines.join('\n'),
    status: 'completed',
  });
}

function emitArtifactsLog(addLog: AddLogFn, result: BridgeResponse): void {
  if (!result.results_path) {
    return;
  }
  addLog({
    agent: 'ArtifactCourier',
    headline: 'Vault transfer',
    content: `Artifacts archived to ${result.results_path}`,
    status: 'completed',
  });
}

function toString(value: unknown): string | undefined {
  if (typeof value === 'string') {
    const trimmed = value.trim();
    return trimmed.length > 0 ? trimmed : undefined;
  }
  return undefined;
}

function toNumber(value: unknown): number | undefined {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === 'string') {
    const parsed = Number(value);
    if (!Number.isNaN(parsed) && Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return undefined;
}

function collectStrings(value: unknown): string[] {
  if (!value) {
    return [];
  }
  if (Array.isArray(value)) {
    return value
      .map((item) => {
        if (typeof item === 'string') {
          return item;
        }
        if (item && typeof item === 'object') {
          const record = item as Record<string, unknown>;
          for (const key of ['text', 'label', 'signal', 'pattern', 'description', 'name', 'summary', 'headline']) {
            const maybe = toString(record[key]);
            if (maybe) {
              return maybe;
            }
          }
        }
        return undefined;
      })
      .filter((line): line is string => Boolean(line));
  }
  if (value && typeof value === 'object') {
    const record = value as Record<string, unknown>;
    return Object.values(record)
      .map((item) => (typeof item === 'string' ? item : undefined))
      .filter((line): line is string => Boolean(line));
  }
  return [];
}

function dedupe(lines: string[]): string[] {
  const seen = new Set<string>();
  const result: string[] = [];
  for (const line of lines) {
    const normalized = line.trim();
    if (!normalized) {
      continue;
    }
    if (!seen.has(normalized)) {
      seen.add(normalized);
      result.push(normalized);
    }
  }
  return result;
}

function buildFundamentalHighlights(fundamental: BridgeFundamental): string[] {
  const highlights: string[] = [];
  const qualitative = fundamental.qualitative_summary as Record<string, unknown> | undefined;
  if (qualitative) {
    const riskFactors = Array.isArray(qualitative.risk_factors)
      ? qualitative.risk_factors
          .map((item) => {
            if (item && typeof item === 'object') {
              const record = item as Record<string, unknown>;
              return toString(record.risk) ?? toString(record.summary);
            }
            return undefined;
          })
          .filter((line): line is string => Boolean(line))
      : [];
    highlights.push(...riskFactors.slice(0, 2));
  }

  const trends = Array.isArray(fundamental.financial_trends)
    ? fundamental.financial_trends
        .map((trend) => {
          if (trend && typeof trend === 'object') {
            const record = trend as Record<string, unknown>;
            const metric = toString(record.metric);
            const direction = toString(record.trend_direction);
            if (metric) {
              return direction ? `${metric}: ${direction}` : metric;
            }
          }
          return undefined;
        })
        .filter((line): line is string => Boolean(line))
    : [];

  highlights.push(...trends.slice(0, 2));

  return dedupe(highlights);
}
