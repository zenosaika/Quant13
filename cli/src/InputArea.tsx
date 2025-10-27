import React, { useState } from 'react';
import { Box, Text } from 'ink';
import TextInput from 'ink-text-input';

export interface InputAreaProps {
  onSubmit: (input: string) => void;
  promptLabel?: string;
}

export const InputArea: React.FC<InputAreaProps> = ({ onSubmit, promptLabel = 'Enter ticker' }) => {
  const [value, setValue] = useState('');

  return (
    <Box paddingX={1} paddingY={0} borderStyle="single" borderColor="gray">
      <Text color="green" bold>
        13▸{' '}
      </Text>
      <TextInput
        value={value}
        placeholder={`${promptLabel}…`}
        onChange={setValue}
        onSubmit={(input) => {
          const trimmed = input.trim();
          if (trimmed.length === 0) {
            return;
          }
          onSubmit(trimmed);
          setValue('');
        }}
      />
    </Box>
  );
};
