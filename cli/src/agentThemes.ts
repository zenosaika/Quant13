export interface AgentPersona {
  label: string;
  accent: string;
  color: string;
  borderColor?: string;
  tagline?: string;
  spinner?: string;
}

export const AGENT_PERSONAS: Record<string, AgentPersona> = {
  system: {
    label: 'Mission Control',
    accent: '◎',
    color: 'cyan',
    borderColor: 'cyan',
    tagline: 'Status uplink from Project 13 core',
  },
  user: {
    label: 'Navigator',
    accent: '✦',
    color: 'white',
    borderColor: 'white',
    tagline: 'Directive received',
  },
  orchestrator: {
    label: 'Orchestrator',
    accent: '∞',
    color: 'blueBright',
    borderColor: 'blue',
    tagline: 'Synchronizing agent collective',
    spinner: 'earth',
  },
  VolatilityModelingAgent: {
    label: 'Volatility Cartographer',
    accent: 'Δσ',
    color: 'cyanBright',
    borderColor: 'cyanBright',
    tagline: 'Mapping implied turbulence contours',
    spinner: 'dots',
  },
  SentimentAgent: {
    label: 'Mood Sensor Array',
    accent: '♫',
    color: 'magentaBright',
    borderColor: 'magenta',
    tagline: 'Amplifying narrative resonance',
    spinner: 'pong',
  },
  TechnicalAnalyst: {
    label: 'Signal Architect',
    accent: '⌁',
    color: 'greenBright',
    borderColor: 'green',
    tagline: 'Reweaving chart harmonics',
    spinner: 'line',
  },
  FundamentalAnalyst: {
    label: 'Balance Sheet Oracle',
    accent: 'Σ',
    color: 'yellowBright',
    borderColor: 'yellow',
    tagline: 'Interpreting corporate DNA',
    spinner: 'dots',
  },
  DebateModerator: {
    label: 'Thesis Arbiter',
    accent: '⚖︎',
    color: 'cyan',
    borderColor: 'cyan',
    tagline: 'Reconciling adversarial hypotheses',
    spinner: 'bouncingBar',
  },
  TraderAgent: {
    label: 'Trade Composer',
    accent: '⚙︎',
    color: 'whiteBright',
    borderColor: 'white',
    tagline: 'Engineering execution schematic',
    spinner: 'arrow3',
  },
  RiskManagementTeam: {
    label: 'Risk Sentinel',
    accent: '⚠︎',
    color: 'redBright',
    borderColor: 'red',
    tagline: 'Guarding capital integrity',
    spinner: 'simpleDotsScrolling',
  },
  ArtifactCourier: {
    label: 'Archive Courier',
    accent: '☍',
    color: 'gray',
    borderColor: 'gray',
    tagline: 'Vaulting artifacts to cold storage',
  },
};

export const getPersona = (agentKey: string): AgentPersona => {
  if (AGENT_PERSONAS[agentKey]) {
    return AGENT_PERSONAS[agentKey];
  }
  return {
    label: agentKey,
    accent: '·',
    color: 'white',
    borderColor: 'gray',
  };
};
