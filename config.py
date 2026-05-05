"""
Trading Bot Configuration
=========================
All settings for the autonomous Binance trading bot.
"""

import os
from typing import Any
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Central configuration for the trading bot."""

    # ──────────────────────────────────────────────
    # BINANCE API
    # ──────────────────────────────────────────────
    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
    BINANCE_API_SECRET: str = os.getenv("BINANCE_API_SECRET", "")
    BINANCE_TESTNET: bool = os.getenv("BINANCE_TESTNET", "true").lower() == "true"

    # ──────────────────────────────────────────────
    # AI BRAIN PROVIDERS
    # ──────────────────────────────────────────────
    AI_PROVIDER: str = os.getenv(
        "AI_PROVIDER",
        (
            "openai"
            if os.getenv("OPENAI_API_KEY")
            else "gemini"
            if os.getenv("GEMINI_API_KEY")
            else "glm"
        ),
    ).lower()

    # OpenAI configuration (primary)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_TEMPERATURE: float = float(
        os.getenv("OPENAI_TEMPERATURE", os.getenv("GLM_TEMPERATURE", "0.3"))
    )
    OPENAI_MAX_TOKENS: int = int(
        os.getenv("OPENAI_MAX_TOKENS", os.getenv("GLM_MAX_TOKENS", "2048"))
    )
    OPENAI_TIMEOUT_SECONDS: int = int(os.getenv("OPENAI_TIMEOUT_SECONDS", "30"))

    # Gemini configuration (primary)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_BASE_URL: str = os.getenv(
        "GEMINI_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta",
    )
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    GEMINI_TEMPERATURE: float = float(
        os.getenv("GEMINI_TEMPERATURE", os.getenv("GLM_TEMPERATURE", "0.3"))
    )
    GEMINI_MAX_TOKENS: int = int(
        os.getenv("GEMINI_MAX_TOKENS", os.getenv("GLM_MAX_TOKENS", "2048"))
    )
    GEMINI_TIMEOUT_SECONDS: int = int(os.getenv("GEMINI_TIMEOUT_SECONDS", "30"))

    # GLM configuration (legacy fallback)
    GLM_API_KEY: str = os.getenv("GLM_API_KEY", "")
    GLM_BASE_URL: str = os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
    GLM_MODEL: str = os.getenv("GLM_MODEL", "glm-5.1")
    GLM_TEMPERATURE: float = float(os.getenv("GLM_TEMPERATURE", "0.3"))
    GLM_MAX_TOKENS: int = int(os.getenv("GLM_MAX_TOKENS", "2048"))

    # ──────────────────────────────────────────────
    # TRADING PARAMETERS
    # ──────────────────────────────────────────────
    # Mode: "spot" or "futures"
    TRADING_MODE: str = os.getenv("TRADING_MODE", "spot")

    # Quote asset to use (e.g., USDT, BUSD, BTC)
    QUOTE_ASSET: str = os.getenv("QUOTE_ASSET", "USDT")

    # Max simultaneous open positions
    MAX_POSITIONS: int = int(os.getenv("MAX_POSITIONS", "5"))

    # Percentage of available balance to use per trade (0.01 = 1%)
    RISK_PER_TRADE: float = float(os.getenv("RISK_PER_TRADE", "0.02"))

    # Stop-loss percentage (0.05 = 5%)
    STOP_LOSS_PCT: float = float(os.getenv("STOP_LOSS_PCT", "0.03"))

    # Take-profit percentage (0.05 = 5%)
    TAKE_PROFIT_PCT: float = float(os.getenv("TAKE_PROFIT_PCT", "0.06"))

    # Trailing stop percentage (0 for disabled)
    TRAILING_STOP_PCT: float = float(os.getenv("TRAILING_STOP_PCT", "0.015"))

    # ──────────────────────────────────────────────
    # TIMEFRAMES (all supported Binance intervals)
    # ──────────────────────────────────────────────
    TIMEFRAMES: list[str] = [
        "1m", "3m", "5m", "15m", "30m",
        "1h", "2h", "4h", "6h", "12h",
        "1d", "3d", "1w", "1M"
    ]

    # Primary timeframes for analysis (used to reduce API calls)
    ANALYSIS_TIMEFRAMES: list[str] = [
        "5m", "15m", "1h", "4h", "1d", "1w"
    ]

    # Timeframe to use for order execution decisions
    EXECUTION_TIMEFRAME: str = "15m"

    # ──────────────────────────────────────────────
    # ASSET FILTERING
    # ──────────────────────────────────────────────
    # Only trade pairs with at least this 24h volume in USDT
    MIN_VOLUME_24H: float = float(os.getenv("MIN_VOLUME_24H", "5_000_000"))

    # Blacklist of symbols to never trade
    BLACKLIST: list[str] = [
        "USDCUSDT", "BUSDUSDT", "TUSDUSDT", "USDPUSDT",
        "FDUSDUSDT", "DAIUSDT", "EURUSDT", "GBPUSDT",
    ]

    # Whitelist (if set, ONLY these symbols will be traded)
    WHITELIST: list[str] | None = None  # e.g., ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    # ──────────────────────────────────────────────
    # AI ANALYSIS SETTINGS
    # ──────────────────────────────────────────────
    # Number of candlesticks to fetch per timeframe
    CANDLES_PER_TIMEFRAME: int = 100

    # Minimum confidence score from GLM to execute a trade (0.0 - 1.0)
    MIN_CONFIDENCE: float = float(os.getenv("MIN_CONFIDENCE", "0.7"))

    # How often to run the analysis loop (seconds)
    ANALYSIS_INTERVAL: int = int(os.getenv("ANALYSIS_INTERVAL", "300"))  # 5 minutes

    # ──────────────────────────────────────────────
    # TECHNICAL INDICATORS
    # ──────────────────────────────────────────────
    INDICATORS: dict = {
        "sma_periods": [20, 50, 100, 200],
        "ema_periods": [9, 21, 55, 200],
        "rsi_period": 14,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "bb_period": 20,
        "bb_std": 2,
        "atr_period": 14,
        "volume_sma": 20,
        "stoch_k": 14,
        "stoch_d": 3,
    }

    # ──────────────────────────────────────────────
    # LOGGING
    # ──────────────────────────────────────────────
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "trading_bot.log")

    # ──────────────────────────────────────────────
    # SAFETY
    # ──────────────────────────────────────────────
    # Max daily loss percentage before halting all trades
    MAX_DAILY_LOSS_PCT: float = float(os.getenv("MAX_DAILY_LOSS_PCT", "0.05"))

    # Enable dry-run mode (no real orders placed)
    DRY_RUN: bool = os.getenv("DRY_RUN", "true").lower() == "true"

    @property
    def AI_MODEL(self) -> str:
        """Return the active model name based on configured provider."""
        if self.AI_PROVIDER == "openai":
            return self.OPENAI_MODEL
        if self.AI_PROVIDER == "gemini":
            return self.GEMINI_MODEL
        return self.GLM_MODEL

    @property
    def ACTIVE_AI_KEY(self) -> str:
        """Return the active provider API key."""
        if self.AI_PROVIDER == "openai":
            return self.OPENAI_API_KEY
        if self.AI_PROVIDER == "gemini":
            return self.GEMINI_API_KEY
        return self.GLM_API_KEY

    @property
    def ACTIVE_AI_KEY_NAME(self) -> str:
        """Return the active provider API key environment variable name."""
        if self.AI_PROVIDER == "openai":
            return "OPENAI_API_KEY"
        if self.AI_PROVIDER == "gemini":
            return "GEMINI_API_KEY"
        return "GLM_API_KEY"


class _Section:
    """Small helper for dot-access config sections."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getattr__(self, name: str) -> Any:
        # Keep compatibility with dynamic config keys expected across modules.
        raise AttributeError(name)


class _CompatConfig:
    """Backward-compatible config object used by strategy and core modules."""

    def __init__(self):
        api_key = os.getenv("BINANCE_API_KEY", "")
        api_secret = os.getenv("BINANCE_API_SECRET", "")

        self.binance = _Section(
            api_key=api_key,
            api_secret=api_secret,
            testnet=os.getenv("BINANCE_TESTNET", "true").lower() == "true",
            testnet_api_key=os.getenv("BINANCE_TESTNET_API_KEY", api_key),
            testnet_secret=os.getenv("BINANCE_TESTNET_SECRET", api_secret),
        )

        self.strategy = _Section(
            primary_tf=os.getenv("PRIMARY_TF", "15m"),
            htf_1=os.getenv("HTF_1", "1h"),
            htf_2=os.getenv("HTF_2", "4h"),
            min_confidence=float(os.getenv("MIN_CONFIDENCE", "0.70")),
            ensemble_min_signals=int(os.getenv("ENSEMBLE_MIN_SIGNALS", "2")),
            supertrend_period=int(os.getenv("SUPERTREND_PERIOD", "10")),
            supertrend_multiplier=float(os.getenv("SUPERTREND_MULTIPLIER", "3.0")),
            rsi_period=int(os.getenv("RSI_PERIOD", "14")),
            rsi_oversold=float(os.getenv("RSI_OVERSOLD", "30")),
            rsi_overbought=float(os.getenv("RSI_OVERBOUGHT", "70")),
            ema_fast=int(os.getenv("EMA_FAST", "9")),
            ema_slow=int(os.getenv("EMA_SLOW", "21")),
            ema_trend=int(os.getenv("EMA_TREND", "50")),
            adx_period=int(os.getenv("ADX_PERIOD", "14")),
            adx_threshold=float(os.getenv("ADX_THRESHOLD", "25")),
            volume_ma_period=int(os.getenv("VOLUME_MA_PERIOD", "20")),
            volume_spike_ratio=float(os.getenv("VOLUME_SPIKE_RATIO", "1.5")),
            breakout_period=int(os.getenv("BREAKOUT_PERIOD", "20")),
        )

        self.risk = _Section(
            initial_capital=float(os.getenv("INITIAL_CAPITAL_USDT", "1000")),
            max_risk_per_trade=float(os.getenv("MAX_RISK_PER_TRADE", "0.015")),
            max_open_positions=int(os.getenv("MAX_OPEN_POSITIONS", "4")),
            max_portfolio_risk=float(os.getenv("MAX_PORTFOLIO_RISK", "0.06")),
            leverage=int(os.getenv("LEVERAGE", "5")),
            atr_sl_multiplier=float(os.getenv("ATR_SL_MULTIPLIER", "1.5")),
            atr_tp1_multiplier=float(os.getenv("ATR_TP1_MULTIPLIER", "2.5")),
            atr_tp2_multiplier=float(os.getenv("ATR_TP2_MULTIPLIER", "4.0")),
            partial_exit_pct=float(os.getenv("PARTIAL_EXIT_PCT", "0.5")),
            min_rr_ratio=float(os.getenv("MIN_RR_RATIO", "1.5")),
        )

        self.trading = _Section(
            candle_limit=int(os.getenv("CANDLE_LIMIT", "250")),
        )


# Primary config object expected by the bot modules.
CONFIG = _CompatConfig()
