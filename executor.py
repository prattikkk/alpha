"""
core/executor.py
Places orders on Binance Futures TESTNET only.
Uses HMAC-signed REST requests — no SDK dependency.
"""
from __future__ import annotations
import hashlib
import hmac
import time
import math
import os
import requests
from typing import Optional
from urllib.parse import urlencode
from core.signal import Direction
from core.risk_manager import PositionSize
from config import CONFIG
from utils.logger import get_logger

log = get_logger("Executor")

TESTNET_BASE = "https://testnet.binancefuture.com"
PAPI_BASE = "https://papi.binance.com"


class TestnetExecutor:
    """
    All order execution happens on Binance Futures TESTNET.
    Market data is NEVER fetched here — see DataFetcher.
    """

    def __init__(self):
        self.api_key    = CONFIG.binance.testnet_api_key
        self.api_secret = CONFIG.binance.testnet_secret
        self.session    = requests.Session()
        self.session.headers.update({
            "X-MBX-APIKEY": self.api_key,
            "Content-Type": "application/json",
        })
        self.papi_base = os.getenv(
            "BINANCE_PAPI_BASE_URL",
            TESTNET_BASE if CONFIG.binance.testnet else PAPI_BASE,
        )
        self._exchange_cache: dict[str, dict] = {}

    # ------------------------------------------------------------------ #
    # Public methods
    # ------------------------------------------------------------------ #

    def open_position(self, pos: PositionSize) -> dict:
        """
        Places:
          1. Market entry order
                    2. Stop-Loss order (UM algo conditional STOP_MARKET)
                    3. TP1 order (UM algo conditional TAKE_PROFIT_MARKET, 50%)
                    4. TP2 order (UM algo conditional TAKE_PROFIT_MARKET, remaining 50%)
        Returns dict of order IDs.
        """
        side       = "BUY"  if pos.direction == Direction.LONG  else "SELL"
        close_side = "SELL" if pos.direction == Direction.LONG  else "BUY"

        # 1. Set leverage
        self._set_leverage(pos.symbol, pos.leverage)

        # 2. Entry — MARKET
        entry_order = self._place_order(
            symbol=pos.symbol,
            side=side,
            order_type="MARKET",
            quantity=pos.quantity,
        )
        if not entry_order:
            log.error(f"Entry order failed for {pos.symbol}")
            return {}

        qty_half = self._round_qty(pos.quantity / 2, pos.symbol)
        qty_rest = self._round_qty(pos.quantity - qty_half, pos.symbol)

        # 3. Stop Loss — algo conditional
        sl_order = self._place_protective_order(
            symbol=pos.symbol,
            side=close_side,
            order_type="STOP_MARKET",
            quantity=pos.quantity,
            trigger_price=self._tick_round(pos.stop_loss, pos.symbol),
        )

        # 4. TP1 — partial algo conditional
        tp1_order = self._place_protective_order(
            symbol=pos.symbol,
            side=close_side,
            order_type="TAKE_PROFIT_MARKET",
            quantity=qty_half,
            trigger_price=self._tick_round(pos.take_profit_1, pos.symbol),
        )

        # 5. TP2 — rest algo conditional
        tp2_order = self._place_protective_order(
            symbol=pos.symbol,
            side=close_side,
            order_type="TAKE_PROFIT_MARKET",
            quantity=qty_rest,
            trigger_price=self._tick_round(pos.take_profit_2, pos.symbol),
        )

        if not sl_order or not tp1_order or not tp2_order:
            log.error(
                f"[{pos.symbol}] protective orders incomplete (SL/TP). Attempting emergency close of entry."
            )
            self._place_order(
                symbol=pos.symbol,
                side=close_side,
                order_type="MARKET",
                quantity=pos.quantity,
            )
            return {}

        ids = {
            "entry": entry_order.get("orderId"),
            "sl":    self._extract_protective_id(sl_order),
            "tp1":   self._extract_protective_id(tp1_order),
            "tp2":   self._extract_protective_id(tp2_order),
        }
        log.info(f"✅ Orders placed [{pos.symbol}]: {ids}")
        return ids

    def cancel_order(self, symbol: str, order_id: int | str) -> bool:
        order_ref = str(order_id)
        if order_ref.startswith("algo:"):
            algo_id = order_ref.split(":", 1)[1]
            return self._cancel_um_algo_order(algo_id)

        if order_ref.startswith("cond:"):
            strategy_id = order_ref.split(":", 1)[1]
            return self._cancel_um_conditional_order(symbol, strategy_id)

        params = {"symbol": symbol, "orderId": order_id}
        resp = self._signed_delete("/fapi/v1/order", params)
        return resp is not None

    def cancel_all_open_orders(self, symbol: str) -> bool:
        params = {"symbol": symbol}
        resp = self._signed_delete("/fapi/v1/allOpenOrders", params)
        return resp is not None

    def get_position(self, symbol: str) -> Optional[dict]:
        """Query current testnet position."""
        resp = self._signed_get("/fapi/v2/positionRisk", {"symbol": symbol})
        if resp and isinstance(resp, list) and len(resp) > 0:
            return resp[0]
        return None

    def get_account(self) -> Optional[dict]:
        return self._signed_get("/fapi/v2/account", {})

    def get_order_status(self, symbol: str, order_id: int) -> Optional[dict]:
        return self._signed_get("/fapi/v1/order", {"symbol": symbol, "orderId": order_id})

    def get_open_positions(self, symbols: Optional[list[str]] = None) -> dict[str, dict]:
        """Get non-zero futures positions keyed by symbol."""
        account = self.get_account()
        if not account:
            return {}

        symbol_filter = set(symbols or [])
        positions: dict[str, dict] = {}
        for pos in account.get("positions", []):
            symbol = pos.get("symbol")
            if not symbol:
                continue
            if symbol_filter and symbol not in symbol_filter:
                continue

            qty = self._to_float(pos.get("positionAmt", 0))
            if abs(qty) <= 0:
                continue

            positions[symbol] = {
                "symbol": symbol,
                "quantity": qty,
                "entry_price": self._to_float(pos.get("entryPrice", 0)),
                "mark_price": self._to_float(pos.get("markPrice", 0)),
                "notional": self._to_float(pos.get("notional", 0)),
                "leverage": int(self._to_float(pos.get("leverage", 1)) or 1),
            }

        return positions

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        stop_price: Optional[float] = None,
        price: Optional[float] = None,
        close_position: bool = False,
    ) -> Optional[dict]:
        params: dict = {
            "symbol":   symbol,
            "side":     side,
            "type":     order_type,
            "quantity": quantity,
        }
        if stop_price:
            params["stopPrice"] = stop_price
        if price:
            params["price"] = price
            params["timeInForce"] = "GTC"
        if close_position:
            params["closePosition"] = "true"
            params.pop("quantity", None)

        resp = self._signed_post("/fapi/v1/order", params)
        if resp:
            log.debug(f"  Order: {order_type} {side} {symbol} qty={quantity} → id={resp.get('orderId')}")
        return resp

    def _set_leverage(self, symbol: str, leverage: int):
        self._signed_post("/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage})

    def _place_protective_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        trigger_price: float,
    ) -> Optional[dict]:
        """Place protective conditional order via UM algo endpoint."""
        qty = self._round_qty(quantity, symbol)
        if qty <= 0:
            return None

        params = {
            "algoType": "CONDITIONAL",
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": qty,
            "triggerPrice": trigger_price,
            "workingType": "CONTRACT_PRICE",
            "priceProtect": "false",
            "reduceOnly": "true",
            "newOrderRespType": "ACK",
        }
        resp = self._signed_post_papi("/papi/v1/um/algo/order", params)
        if resp:
            return resp

        # Fallback for accounts that only support legacy conditional endpoint.
        fallback = {
            "symbol": symbol,
            "side": side,
            "strategyType": order_type,
            "quantity": qty,
            "stopPrice": trigger_price,
            "workingType": "CONTRACT_PRICE",
            "priceProtect": "false",
            "reduceOnly": "true",
        }
        return self._signed_post_papi("/papi/v1/um/conditional/order", fallback)

    def _cancel_um_algo_order(self, algo_id: str) -> bool:
        resp = self._signed_delete_papi("/papi/v1/um/algo/order", {"algoId": algo_id})
        if isinstance(resp, dict) and "complete" in resp:
            return bool(resp.get("complete"))
        return resp is not None

    def _cancel_um_conditional_order(self, symbol: str, strategy_id: str) -> bool:
        params = {
            "symbol": symbol,
            "strategyId": strategy_id,
        }
        resp = self._signed_delete_papi("/papi/v1/um/conditional/order", params)
        return resp is not None

    @staticmethod
    def _extract_protective_id(order: Optional[dict]) -> Optional[str | int]:
        if not order:
            return None
        if "algoId" in order:
            return f"algo:{order['algoId']}"
        if "strategyId" in order:
            return f"cond:{order['strategyId']}"
        if "orderId" in order:
            return order["orderId"]
        return None

    def _signed_post(self, path: str, params: dict) -> Optional[dict]:
        return self._signed_request("POST", path, params, TESTNET_BASE)

    def _signed_post_papi(self, path: str, params: dict) -> Optional[dict]:
        return self._signed_request("POST", path, params, self.papi_base)

    def _signed_get(self, path: str, params: dict) -> Optional[dict]:
        return self._signed_request("GET", path, params, TESTNET_BASE)

    def _signed_delete(self, path: str, params: dict) -> Optional[dict]:
        return self._signed_request("DELETE", path, params, TESTNET_BASE)

    def _signed_delete_papi(self, path: str, params: dict) -> Optional[dict]:
        return self._signed_request("DELETE", path, params, self.papi_base)

    def _signed_request(
        self,
        method: str,
        path: str,
        params: dict,
        base_url: str,
    ) -> Optional[dict]:
        if not self.api_key or not self.api_secret:
            log.warning("Testnet API keys not configured — order skipped (dry run mode)")
            return {"orderId": 0, "status": "DRY_RUN"}

        params["timestamp"] = int(time.time() * 1000)
        query = urlencode(params)
        sig = hmac.new(
            self.api_secret.encode(), query.encode(), hashlib.sha256
        ).hexdigest()
        query += f"&signature={sig}"

        url = f"{base_url}{path}"
        try:
            if method == "POST":
                resp = self.session.post(f"{url}?{query}", timeout=10)
            elif method == "DELETE":
                resp = self.session.delete(f"{url}?{query}", timeout=10)
            else:
                resp = self.session.get(f"{url}?{query}", timeout=10)

            if resp.status_code in (200, 201):
                return resp.json()
            else:
                text = resp.text[:200].replace("\n", " ") if resp.text else ""
                log.error(
                    f"API {method} {base_url}{path} → {resp.status_code}: {text}"
                )
                return None
        except Exception as e:
            log.error(f"Testnet request failed [{path}]: {e}")
            return None

    @staticmethod
    def _to_float(value, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _round_qty(self, qty: float, symbol: str) -> float:
        step = self._get_exchange_info(symbol).get("step_size", 0.001)
        if step <= 0:
            return round(qty, 3)
        precision = max(0, -int(math.floor(math.log10(step))))
        factor = 10 ** precision
        return math.floor(qty * factor) / factor

    def _tick_round(self, price: float, symbol: str) -> float:
        tick = self._get_exchange_info(symbol).get("tick_size", 0.01)
        if tick <= 0:
            return round(price, 2)
        precision = max(0, -int(math.floor(math.log10(tick))))
        return round(round(price / tick) * tick, precision)

    def _get_exchange_info(self, symbol: str) -> dict:
        if symbol in self._exchange_cache:
            return self._exchange_cache[symbol]
        # Attempt live fetch from testnet
        try:
            r = self.session.get(
                f"{TESTNET_BASE}/fapi/v1/exchangeInfo",
                timeout=10
            )
            if r.status_code == 200:
                for s in r.json().get("symbols", []):
                    if s["symbol"] == symbol:
                        info = {}
                        for f in s.get("filters", []):
                            ft = f["filterType"]
                            if ft == "PRICE_FILTER":
                                info["tick_size"] = float(f["tickSize"])
                            elif ft == "LOT_SIZE":
                                info["step_size"] = float(f["stepSize"])
                                info["min_qty"] = float(f["minQty"])
                        self._exchange_cache[symbol] = info
                        return info
        except Exception:
            pass
        return {"tick_size": 0.01, "step_size": 0.001, "min_qty": 0.001}
