# 🤖 AlphaBot — Binance Futures Paper Trading Bot

**Fetches real market data from Binance MAINNET → Executes paper trades on TESTNET**

---

## Architecture

```
Binance MAINNET (market data, read-only)
        │  OHLCV candles
        ▼
  DataFetcher
        │
        ▼
  Strategy Engine ──────────────────────────────────
  │  SuperTrend + RSI    (weight: 45%)              │
  │  EMA + ADX + Volume  (weight: 35%)              │
  │  Breakout Momentum   (weight: 20%)              │
  └────────────── Ensemble: 2/3 must agree ─────────
        │  Signal (direction, confidence, SL/TP)
        ▼
  Risk Manager (Kelly-fraction position sizing)
        │  PositionSize
        ▼
  Testnet Executor ← Binance Futures TESTNET
        │  Market entry + SL + TP1 + TP2 orders
        ▼
  Portfolio Tracker (P&L, drawdown, stats)
        │
        ▼
  Telegram Alerts (optional)
```

---

## Strategies

### 1. SuperTrend + RSI (Primary)
- SuperTrend flip signals trend change
- RSI confirms momentum (not overbought/oversold)
- 1h/4h EMA alignment for HTF bias
- Best in: trending markets

### 2. EMA + ADX + Volume
- 3-EMA stack (9/21/50) confirms trend direction
- ADX > 25 ensures we're in a trend (not ranging)
- Volume spike (1.5×) confirms conviction
- Pullback to slow EMA avoids chasing
- Best in: strong directional moves

### 3. Breakout Momentum
- Dual breakout: Bollinger Band + Donchian Channel
- RSI > 52 / < 48 for momentum confirmation
- Volume spike (1.8×) confirms the break
- ATR expanding = volatility increasing
- Best in: volatile breakout sessions

### Ensemble (Default)
- Aggregates all three strategies
- Trade only when ≥ 2 strategies agree on direction
- Weighted confidence scoring
- Highest-quality signal filtering

---

## Risk Management
- **Position sizing**: Fixed fractional (1.5% risk per trade)
- **Leverage**: 5× (conservative)
- **SL**: 1.5× ATR below/above entry
- **TP1**: 2.5× ATR — 50% partial exit
- **TP2**: 4.0× ATR — remaining 50%
- **Trailing stop**: After TP1 hit, SL moves to breakeven
- **Max positions**: 4 concurrent
- **Max portfolio risk**: 6% total

---

## Setup

### 1. Clone & install
```bash
git clone <repo>
cd alphabot
pip install -r requirements.txt
```

### 2. Configure
```bash
cp .env.example .env
# Edit .env with your API keys
```

Get testnet keys: https://testnet.binancefuture.com/
- Mainnet keys: for market data only (read-only)
- Testnet keys: for paper trade order execution

### 3. Backtest first (strongly recommended)
```bash
# Validate on 60 days of BTC 15m data
python backtest.py --symbol BTCUSDT --tf 15m --days 60 --strategy ensemble

# Try other pairs
python backtest.py --symbol ETHUSDT --tf 15m --days 90
python backtest.py --symbol SOLUSDT --tf 15m --days 60 --strategy supertrend_rsi
```

### 4. Dry run (no orders placed)
```bash
DRY_RUN=true python main.py
```

### 5. Live paper trading
```bash
python main.py
# or with Docker:
docker-compose up -d
docker-compose logs -f
```

---

## File Structure
```
alphabot/
├── main.py                    # Bot orchestrator
├── backtest.py                # Historical backtester
├── config.py                  # Centralised config
├── .env.example               # Environment template
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── core/
│   ├── data_fetcher.py        # Mainnet OHLCV (read-only)
│   ├── indicators.py          # Pure numpy/pandas indicators
│   ├── signal.py              # Signal dataclass
│   ├── risk_manager.py        # Position sizing
│   ├── portfolio.py           # P&L tracking
│   ├── executor.py            # Testnet order execution
│   └── position_monitor.py   # Exit management
├── strategies/
│   ├── supertrend_rsi.py
│   ├── ema_adx_volume.py
│   ├── breakout_momentum.py
│   └── ensemble.py            # Meta-strategy
├── utils/
│   ├── logger.py
│   └── notifier.py            # Telegram alerts
├── logs/                      # Auto-created
└── data/                      # Auto-created (portfolio.json, backtest CSVs)
```

---

## Key Config Options (.env)

| Variable | Default | Description |
|---|---|---|
| `ACTIVE_STRATEGY` | `ensemble` | Strategy to use |
| `SYMBOLS` | `BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT` | Pairs to trade |
| `INITIAL_CAPITAL_USDT` | `1000` | Starting paper capital |
| `MAX_RISK_PER_TRADE` | `0.015` | 1.5% risk per trade |
| `MAX_OPEN_POSITIONS` | `4` | Max concurrent positions |
| `PRIMARY_TF` | `15m` | Entry timeframe |
| `HTF_1` | `1h` | Higher timeframe 1 |
| `HTF_2` | `4h` | Higher timeframe 2 |
| `DRY_RUN` | `false` | Skip order execution |

---

## ⚠️ Important Notes

1. **This is paper trading** — all orders go to testnet, no real money at risk
2. **Past performance ≠ future results** — always backtest before running
3. **Crypto is volatile** — even good strategies have drawdowns
4. **Backtest ≥ 30 trades** before evaluating a strategy
5. **Never commit API keys** to git (`.env` is gitignored)

---

## Deployment (Oracle Cloud Free Tier — 24/7)
```bash
# On your Oracle VM:
git clone <repo> && cd alphabot
cp .env.example .env && nano .env   # add your keys
docker-compose up -d
```
