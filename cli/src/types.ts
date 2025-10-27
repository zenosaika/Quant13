export type AgentStatus = 'pending' | 'completed' | 'error';

export interface LogMessage {
  id: string;
  agent: string;
  content: string;
  status: AgentStatus;
  headline?: string;
  metadata?: Record<string, unknown>;
}

export type AddLogFn = (log: Omit<LogMessage, 'id'>) => string;
export type UpdateLogFn = (id: string, updates: Partial<LogMessage>) => void;
