import React from 'react';
import { Box, Text } from 'ink';
import Spinner from 'ink-spinner';
import type { SpinnerName } from 'cli-spinners';
import type { LogMessage } from './types.js';
import { getPersona } from './agentThemes.js';

const gaugeCharacters = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

const createGauge = (percent: number, length = 16): string => {
  if (Number.isNaN(percent)) {
    return ''.padEnd(length, '─');
  }
  const normalized = Math.max(0, Math.min(100, percent));
  const blocks = Math.round((normalized / 100) * length);
  const filled = gaugeCharacters[gaugeCharacters.length - 1] ?? '█';
  return `${filled.repeat(blocks)}${' '.repeat(Math.max(0, length - blocks))}`;
};

const formatLines = (content: string | undefined): string[] => {
  if (!content) {
    return [];
  }
  return content.split('\n').filter((line) => line.trim().length > 0);
};

const renderVolatility = (log: LogMessage): React.ReactNode => {
  const metadata = (log.metadata ?? {}) as Record<string, unknown>;
  const ivRank = typeof metadata.ivRank === 'number' ? metadata.ivRank : undefined;
  const forecast = typeof metadata.forecast === 'string' ? metadata.forecast : log.content;
  const skew = typeof metadata.skew === 'string' ? metadata.skew : undefined;
  const term = typeof metadata.term === 'string' ? metadata.term : undefined;
  const gauge = ivRank !== undefined ? createGauge(ivRank) : undefined;

  return (
    <Box flexDirection="column" gap={1}>
      {ivRank !== undefined && (
        <Text>
          IV Rank {ivRank.toFixed(0).padStart(2, ' ')} |{gauge}| {ivRank >= 60 ? '⚠︎' : ivRank <= 40 ? '◎' : '•'}
        </Text>
      )}
      <Text>{forecast}</Text>
      {skew && <Text color="cyan">{skew}</Text>}
      {term && <Text color="cyan">{term}</Text>}
    </Box>
  );
};

const renderSentiment = (log: LogMessage): React.ReactNode => {
  const metadata = (log.metadata ?? {}) as Record<string, unknown>;
  const score = typeof metadata.score === 'number' ? metadata.score : undefined;
  const summary = typeof metadata.summary === 'string' ? metadata.summary : log.content;
  const spotlight = metadata.spotlight as Record<string, unknown> | undefined;
  const normalizedScore = score !== undefined ? Math.max(0, Math.min(1, (score + 1) / 2)) : undefined;
  const gauge = normalizedScore !== undefined ? createGauge(normalizedScore * 100) : undefined;

  return (
    <Box flexDirection="column" gap={1}>
      {score !== undefined && (
        <Text>
          Sentiment {score.toFixed(2)} |{gauge}| {score > 0.25 ? '🌈' : score < -0.25 ? '☁️' : '⚖︎'}
        </Text>
      )}
      <Text>{summary}</Text>
      {spotlight && (
        <Text color="magenta">
          Spotlight ▸ {typeof spotlight.title === 'string' ? spotlight.title : 'Top headline'}
          {typeof spotlight.publisher === 'string' ? ` (${spotlight.publisher})` : ''}
        </Text>
      )}
    </Box>
  );
};

const renderTechnical = (log: LogMessage): React.ReactNode => {
  const metadata = (log.metadata ?? {}) as Record<string, unknown>;
  const bias = typeof metadata.bias === 'string' ? metadata.bias : undefined;
  const highlights = Array.isArray(metadata.highlights) ? metadata.highlights : [];
  const summary = typeof metadata.summary === 'string' ? metadata.summary : log.content;

  return (
    <Box flexDirection="column" gap={1}>
      {bias && <Text color="green">Primary Bias ▸ {bias}</Text>}
      <Text>{summary}</Text>
      {highlights.slice(0, 3).map((item, index) => (
        <Text key={index} color="green">
          • {item}
        </Text>
      ))}
    </Box>
  );
};

const renderFundamental = (log: LogMessage): React.ReactNode => {
  const metadata = (log.metadata ?? {}) as Record<string, unknown>;
  const summary = typeof metadata.summary === 'string' ? metadata.summary : log.content;
  const ratios = (metadata.ratios as Record<string, number | null> | undefined) ?? {};
  const ratiosLine = Object.entries(ratios)
    .slice(0, 3)
    .map(([key, value]) => `${key}: ${value != null ? value.toFixed(2) : '—'}`)
    .join(' | ');
  const highlights = Array.isArray(metadata.highlights) ? metadata.highlights : [];

  return (
    <Box flexDirection="column" gap={1}>
      <Text>{summary}</Text>
      {ratiosLine && <Text color="yellow">{ratiosLine}</Text>}
      {highlights.slice(0, 2).map((line, index) => (
        <Text key={index} color="yellow">
          • {line}
        </Text>
      ))}
    </Box>
  );
};

const renderDefault = (log: LogMessage): React.ReactNode => (
  <Box flexDirection="column">
    {formatLines(log.content).map((line, index) => (
      <Text key={index}>{line}</Text>
    ))}
  </Box>
);

const renderCompleted = (log: LogMessage): React.ReactNode => {
  switch (log.agent) {
    case 'VolatilityModelingAgent':
      return renderVolatility(log);
    case 'SentimentAgent':
      return renderSentiment(log);
    case 'TechnicalAnalyst':
      return renderTechnical(log);
    case 'FundamentalAnalyst':
      return renderFundamental(log);
    default:
      return renderDefault(log);
  }
};

const LogEntry: React.FC<{ log: LogMessage }> = ({ log }) => {
  const persona = getPersona(log.agent);
  const spinnerName = (persona.spinner ?? 'dots') as SpinnerName;
  const borderColor = persona.borderColor ?? persona.color ?? 'white';

  return (
    <Box
      marginBottom={1}
      flexDirection="column"
      borderStyle="round"
      borderColor={borderColor}
      paddingX={1}
      paddingY={0}
      gap={1}
    >
      <Box flexDirection="row" justifyContent="space-between" flexWrap="wrap">
        <Text color={persona.color} bold>
          {persona.accent} {persona.label}
        </Text>
        {persona.tagline && (
          <Text color="gray">{persona.tagline}</Text>
        )}
      </Box>

      {log.headline && log.status === 'completed' && (
        <Text color={persona.color} bold>
          {log.headline}
        </Text>
      )}

      {log.status === 'pending' && (
        <Text color={persona.color}>
          <Spinner type={spinnerName} /> {log.content || 'Processing...'}
        </Text>
      )}

      {log.status === 'completed' && renderCompleted(log)}

      {log.status === 'error' && (
        <Text color="red">⚠︎ {log.content}</Text>
      )}
    </Box>
  );
};

export interface LogAreaProps {
  logs: LogMessage[];
}

export const LogArea: React.FC<LogAreaProps> = ({ logs }) => (
  <Box flexGrow={1} flexDirection="column" padding={1} overflowY="hidden">
    <Box flexGrow={1} />
    {logs.map((log) => (
      <LogEntry key={log.id} log={log} />
    ))}
  </Box>
);
