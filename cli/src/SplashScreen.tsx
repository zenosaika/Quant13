import React from 'react';
import { Box, Text } from 'ink';
import Gradient from 'ink-gradient';
import BigText from 'ink-big-text';
import Spinner from 'ink-spinner';

export interface SplashScreenProps {
  isInteractive: boolean;
}

export const SplashScreen: React.FC<SplashScreenProps> = ({ isInteractive }) => (
  <Box
    flexDirection="column"
    alignItems="center"
    justifyContent="center"
    height="100%"
    width="100%"
    gap={1}
    padding={1}
  >
    <Gradient name="pastel">
      <BigText text="Project 13" font="simple" />
    </Gradient>
    <Text color="cyan">Quantum Market Orchestration Console</Text>
    <Text color="gray">Codename: Project 13 // Initiative: Synthesise + Execute</Text>
    <Text color="magenta">
      <Spinner type="dots" /> {isInteractive ? 'Link established. Awaiting navigator command…' : 'Listening for piped directives…'}
    </Text>
  </Box>
);
