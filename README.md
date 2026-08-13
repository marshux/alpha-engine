# alpha-engine

Systematic swing-trading research engine, aiming to beat SPY/VOO/VTI over
days-to-weeks holding periods. Quant signals (momentum, mean-reversion,
volatility) plus an AI-powered news research signal feed into a fully
deterministic, rules-based decision engine — the AI researches and scores,
it never decides. See `CLAUDE.md` for the full architecture writeup.

Currently backtest-only: no live or paper order execution yet.

## Setup

```
python3 -m venv .venv   # see CLAUDE.md if this fails (no system pip/venv)
.venv/bin/pip install -e ".[dev]"
cp .env.example .env    # fill in ANTHROPIC_API_KEY (only needed for the live news signal)
```

## Run a backtest

```
.venv/bin/python3 -m alpha_engine.cli backtest --start 2019-01-01 --end 2024-12-31
```

## Run tests

```
.venv/bin/python3 -m pytest tests/ -v
```
