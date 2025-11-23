import React, { useMemo } from 'react';
import { Box, Text } from 'ink';
import Gradient from 'ink-gradient';
import Spinner from 'ink-spinner';

export type HeaderProps = {
  appName: string;
  status: string;
  activeTicker?: string;
};

export const Header: React.FC<HeaderProps> = ({ appName, status, activeTicker }) => {
  const statusConfig = useMemo(() => {
    const normalized = status.toLowerCase();
    if (normalized.includes('running') || normalized.includes('analysis')) {
      return { color: 'yellow', label: status, active: true };
    }
    if (normalized.includes('error')) {
      return { color: 'red', label: status, active: false };
    }
    return { color: 'green', label: status, active: false };
  }, [status]);

  return (
    <Box
      borderStyle="round"
      borderColor="gray"
      paddingX={1}
      paddingY={0}
      flexDirection="row"
      justifyContent="space-between"
      alignItems="center"
      width="100%"
    >
      <Box flexDirection="column" flexGrow={1} gap={0}>
        <Gradient name="atlas">
          <Text bold>{appName}</Text>
        </Gradient>
        <Text color={statusConfig.color}>
          {statusConfig.active ? (
            <Text>
              <Spinner type="dots" /> {statusConfig.label}
            </Text>
          ) : (
            `● ${statusConfig.label}`
          )}
        </Text>
      </Box>

      <Box flexDirection="column" alignItems="flex-end" paddingLeft={2}>
        <Text color="cyan">
          {activeTicker ? `Focus ▸ ${activeTicker.toUpperCase()}` : 'Awaiting directive'}
        </Text>
        <Text color="gray">{process.cwd()}</Text>
      </Box>
    </Box>
  );
};
