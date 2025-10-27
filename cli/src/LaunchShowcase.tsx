import React, { useEffect, useMemo, useState } from 'react';
import { Box, Text } from 'ink';
import Gradient from 'ink-gradient';
import BigText from 'ink-big-text';

const colorCycles = [
  { gradient: ['#0ea5e9', '#9333ea'], accent: '#38bdf8' },
  { gradient: ['#22d3ee', '#8b5cf6'], accent: '#34d5ff' },
  { gradient: ['#2dd4bf', '#6366f1'], accent: '#5eead4' }
];

const taglines = [
  'Synchronising predictive lattice…',
  'Mapping liquidity vortices…',
  'Sequencing risk sentinels…'
];

const symbols = ['⌁', '✶', '⚡', '⟡'];

const wave = '▁▂▃▄▅▆▇█▇▆▅▄▃▂▁';

const FRAME_WIDTH = 30;

const centerLine = (content: string): string => {
  const normalized = content.length > FRAME_WIDTH ? content.slice(0, FRAME_WIDTH) : content;
  const remaining = FRAME_WIDTH - normalized.length;
  const padStart = Math.floor(remaining / 2);
  const padEnd = remaining - padStart;
  return `┃${' '.repeat(padStart)}${normalized}${' '.repeat(padEnd)}┃`;
};

const buildFrame = (symbol: string): string[] => [
  `┏${'━'.repeat(FRAME_WIDTH)}┓`,
  centerLine(`Q U A N T   1 3   ${symbol}`),
  centerLine('Signal lattice primed & live'),
  `┗${'━'.repeat(FRAME_WIDTH)}┛`
];

export const LaunchShowcase: React.FC = () => {
  const [paletteIndex, setPaletteIndex] = useState(0);
  const [symbolIndex, setSymbolIndex] = useState(0);
  const [taglineIndex, setTaglineIndex] = useState(0);
  const [waveOffset, setWaveOffset] = useState(0);

  useEffect(() => {
    const paletteTimer = setInterval(() => {
      setPaletteIndex((prev) => (prev + 1) % colorCycles.length);
      setSymbolIndex((prev) => (prev + 1) % symbols.length);
      setTaglineIndex((prev) => (prev + 1) % taglines.length);
    }, 1600);

    const waveTimer = setInterval(() => {
      setWaveOffset((prev) => (prev + 1) % wave.length);
    }, 120);

    return () => {
      clearInterval(paletteTimer);
      clearInterval(waveTimer);
    };
  }, []);

  const palette = colorCycles[paletteIndex % colorCycles.length]!;
  const symbol = symbols[symbolIndex % symbols.length]!;
  const asciiFrame = useMemo(() => buildFrame(symbol), [symbol]);
  const waveLine = useMemo(
    () => wave.slice(waveOffset) + wave.slice(0, waveOffset),
    [waveOffset]
  );

  return (
    <Box flexDirection="column" alignItems="center" gap={1} paddingY={1} flexShrink={0}>
      <Gradient colors={palette.gradient}>
        <BigText text="Quant 13" font="chrome" />
      </Gradient>
      <Box flexDirection="column">
        {asciiFrame.map((line, index) => (
          <Text key={index} color={palette.accent}>
            {line}
          </Text>
        ))}
      </Box>
      <Text color={palette.accent}>{waveLine}</Text>
      <Text color="cyanBright">{taglines[taglineIndex % taglines.length]!}</Text>
    </Box>
  );
};
