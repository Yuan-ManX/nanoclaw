# ClawAI Skills System

This directory contains **built-in skills** that extend **ClawAI Agent Runtime** capabilities.

Skills are **first-class runtime modules** that define:

- Tool usage protocols
- Multi-step reasoning workflows
- External system interaction patterns
- Agent behavior constraints and guidance

They are loaded dynamically by **ToolChain** and **AgentLoop** at runtime.


## Design Philosophy

ClawAI skills follow three core principles:

### 1. Declarative over Imperative

Skills describe **what the agent should do**, not how to implement it.

### 2. Composable Workflows

Skills are composable through **ToolChain**, enabling complex multi-step pipelines.

### 3. Runtime Hot Reload

Skills can be added, removed, or modified **without restarting the runtime**.

