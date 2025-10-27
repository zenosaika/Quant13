import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { render, Box, Text, useApp } from 'ink';
import crypto from 'crypto';
import { Header } from './Header.js';
import type { HeaderProps } from './Header.js';
import { LogArea } from './LogArea.js';
import { InputArea } from './InputArea.js';
import { SplashScreen } from './SplashScreen.js';
import { LaunchShowcase } from './LaunchShowcase.js';
import type { AddLogFn, LogMessage, UpdateLogFn } from './types.js';
import { runAgentLogic } from './agent-logic.js';

const APP_NAME = '🧠 Quant CLI';
const isInteractiveSession = Boolean(process.stdin.isTTY);

const App: React.FC = () => {
  const [logs, setLogs] = useState<LogMessage[]>([]);
  const [status, setStatus] = useState<string>('Idle');
  const [activeTicker, setActiveTicker] = useState<string | undefined>(undefined);
  const [hasAutoSubmitted, setHasAutoSubmitted] = useState<boolean>(false);
  const [showSplash, setShowSplash] = useState<boolean>(true);
  const { exit } = useApp();

  const scheduleExit = useCallback(() => {
    if (isInteractiveSession) {
      return;
    }
    setTimeout(() => {
      exit();
    }, 400);
  }, [exit]);

  const addLog: AddLogFn = useCallback((log) => {
    const id = crypto.randomUUID();
    setLogs((prev) => [...prev, { ...log, id }]);
    return id;
  }, []);

  const updateLog: UpdateLogFn = useCallback((id, updates) => {
    setLogs((prev) =>
      prev.map((log) => (log.id === id ? { ...log, ...updates } : log))
    );
  }, []);

  const handleUserInput = useCallback(
    (input: string) => {
      setActiveTicker(input);
      setStatus('Running analysis');

      addLog({ agent: 'user', content: input, status: 'completed' });

      runAgentLogic(input, addLog, updateLog)
        .catch((error: unknown) => {
          addLog({
            agent: 'system',
            content:
              error instanceof Error
                ? error.message
                : 'Unexpected error occurred during analysis.',
            status: 'error',
          });
        })
        .finally(() => {
          setStatus('Idle');
          scheduleExit();
        });
    },
    [addLog, updateLog, scheduleExit]
  );

  useEffect(() => {
    const splashDuration = isInteractiveSession ? 1400 : 400;
    const timer = setTimeout(() => setShowSplash(false), splashDuration);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    if (isInteractiveSession || hasAutoSubmitted) {
      return;
    }

    process.stdin.setEncoding('utf-8');

    let buffer = '';

    const handleData = (chunk: string) => {
      buffer += chunk;
    };

    const handleEnd = () => {
      if (hasAutoSubmitted) {
        return;
      }
      const candidate = buffer
        .split(/\r?\n/)
        .map((line) => line.trim())
        .find((line) => line.length > 0);

      if (candidate) {
        setHasAutoSubmitted(true);
        handleUserInput(candidate);
      } else {
        addLog({
          agent: 'system',
          content: 'No ticker provided in stdin input.',
          status: 'error',
        });
        scheduleExit();
      }
    };

    process.stdin.on('data', handleData);
    process.stdin.once('end', handleEnd);
    process.stdin.resume();

    return () => {
      process.stdin.off('data', handleData);
      process.stdin.off('end', handleEnd);
    };
  }, [addLog, handleUserInput, hasAutoSubmitted, scheduleExit]);

  const headerProps: HeaderProps = useMemo(() => {
    return activeTicker
      ? { appName: APP_NAME, status, activeTicker }
      : { appName: APP_NAME, status };
  }, [activeTicker, status]);

  if (showSplash) {
    return <SplashScreen isInteractive={isInteractiveSession} />;
  }

  return (
    <Box flexDirection="column" width="100%" height="100%">
      <LaunchShowcase />
      <Header {...headerProps} />
      <LogArea logs={logs} />
      {isInteractiveSession ? (
        <InputArea onSubmit={handleUserInput} promptLabel="Analyze ticker" />
      ) : (
        <Box paddingX={1} paddingY={0} borderStyle="single" borderColor="gray">
          <Text color="gray">Telemetry uplink active; awaiting piped directive…</Text>
        </Box>
      )}
    </Box>
  );
};

render(<App />);

export { App };
