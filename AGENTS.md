# PITS Agent System

PITS operates as a hierarchy of collaborative AI agents, each with a specific domain of expertise.

## Agent Roles

### Asset Specialists
Dedicated to individual assets (e.g., WTI Agent, Gold Agent).
- Focus: Understanding asset-specific micro-patterns and news sensitivity.

### Correlation Architect
Monitors the "web" of assets.
- Focus: Cross-asset lead-lag relationships and global macro shifts (DXY/VIX).

### Risk Auditor
The "CFO" of the system.
- Focus: Ensuring no trade or portfolio exposure violates the risk parameters.

### Execution Strategist
The "Floor Trader."
- Focus: Finding the best entry/exit price and managing order splits.

### Learning Guardian
The "Scientist."
- Focus: Monitoring model performance and triggering recalibration or online learning sessions.

## Communication Protocol
Agents communicate via standardized message formats (JSON) facilitated by the `Brain` orchestrator.
- **Signals**: Probability and confidence scores.
- **Commands**: Requests for execution or risk checks.
- **Health**: Heartbeats and latency reports.
