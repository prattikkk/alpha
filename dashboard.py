"""Lightweight local dashboard for portfolio monitoring."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from config import CONFIG

PORTFOLIO_PATH = Path("data/portfolio.json")


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api/state"):
            self._send_json(_load_state())
            return
        self._send_html(_render_html())

    def log_message(self, format, *args):
        return

    def _send_json(self, payload: dict):
        encoded = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_html(self, html: str):
        encoded = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def _load_state() -> dict:
    if not PORTFOLIO_PATH.exists():
        return {"balance": 0.0, "open_positions": {}, "closed_trades": []}
    try:
        return json.loads(PORTFOLIO_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"balance": 0.0, "open_positions": {}, "closed_trades": []}


def _render_html() -> str:
    return """
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>AlphaBot Dashboard</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500&display=swap');

    :root {
      --bg-top: #f6f1e9;
      --bg-bottom: #e7f1f8;
      --panel: rgba(255, 255, 255, 0.78);
      --panel-strong: rgba(255, 255, 255, 0.9);
      --text: #1f2a37;
      --muted: #55657a;
      --line: #d9e3ef;
      --accent: #05668d;
      --accent-soft: #7ec8e3;
      --profit: #0d9488;
      --loss: #b91c1c;
      --shadow: 0 14px 40px rgba(18, 46, 71, 0.12);
      --radius: 18px;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      color: var(--text);
      font-family: 'Space Grotesk', 'Trebuchet MS', sans-serif;
      background:
        radial-gradient(circle at 10% 15%, #ffe8cf 0%, transparent 35%),
        radial-gradient(circle at 85% 25%, #d7f0ff 0%, transparent 40%),
        linear-gradient(180deg, var(--bg-top), var(--bg-bottom));
      min-height: 100vh;
      animation: pageFade 420ms ease-out;
    }

    .wrap {
      max-width: 1120px;
      margin: 0 auto;
      padding: 22px 16px 30px;
    }

    .hero {
      background: linear-gradient(120deg, rgba(5, 102, 141, 0.1), rgba(126, 200, 227, 0.18));
      border: 1px solid rgba(5, 102, 141, 0.15);
      border-radius: var(--radius);
      padding: 18px 18px 16px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(6px);
      margin-bottom: 14px;
    }

    .title {
      margin: 0;
      font-size: clamp(1.25rem, 4.6vw, 2rem);
      letter-spacing: 0.01em;
      font-weight: 700;
      color: #15324a;
    }

    .sub {
      margin-top: 4px;
      color: var(--muted);
      font-size: 0.95rem;
    }

    .cards {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 12px;
      margin-bottom: 14px;
    }

    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px 14px;
      box-shadow: var(--shadow);
      animation: rise 440ms ease-out both;
    }

    .metric {
      color: var(--muted);
      font-size: 0.86rem;
      letter-spacing: 0.03em;
      text-transform: uppercase;
    }

    .value {
      display: block;
      font-size: 1.4rem;
      font-family: 'IBM Plex Mono', Consolas, monospace;
      margin-top: 6px;
      color: #11344f;
    }

    .panel {
      background: var(--panel-strong);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 12px;
      margin-bottom: 14px;
      overflow: hidden;
    }

    h2 {
      margin: 4px 6px 10px;
      font-size: 1rem;
      color: #15324a;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.92rem;
    }

    th, td {
      text-align: left;
      padding: 9px 8px;
      border-bottom: 1px solid var(--line);
      white-space: nowrap;
    }

    th {
      color: #365069;
      font-weight: 600;
      font-size: 0.8rem;
      letter-spacing: 0.03em;
      text-transform: uppercase;
    }

    tr:last-child td { border-bottom: none; }

    .pnl-pos { color: var(--profit); font-weight: 600; }
    .pnl-neg { color: var(--loss); font-weight: 600; }

    @media (max-width: 740px) {
      .wrap { padding: 14px 10px 18px; }
      .panel { overflow-x: auto; }
      th, td { padding: 8px 7px; }
    }

    @keyframes pageFade {
      from { opacity: 0; }
      to { opacity: 1; }
    }

    @keyframes rise {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"hero\">
      <h1 class=\"title\">AlphaBot Portfolio Radar</h1>
      <div class=\"sub\">Live snapshot of open exposure and latest closed trades.</div>
    </div>

    <div class=\"cards\">
      <div class=\"card\"><div class=\"metric\">Balance</div><strong class=\"value\" id=\"balance\">-</strong></div>
      <div class=\"card\"><div class=\"metric\">Open Positions</div><strong class=\"value\" id=\"openCount\">-</strong></div>
      <div class=\"card\"><div class=\"metric\">Closed Trades</div><strong class=\"value\" id=\"closedCount\">-</strong></div>
    </div>

    <div class=\"panel\">
      <h2>Open Positions</h2>
      <table>
        <thead><tr><th>Symbol</th><th>Direction</th><th>Entry</th><th>SL</th><th>Qty</th><th>PnL</th></tr></thead>
        <tbody id=\"openRows\"></tbody>
      </table>
    </div>

    <div class=\"panel\">
      <h2>Recent Closed Trades</h2>
      <table>
        <thead><tr><th>Symbol</th><th>Reason</th><th>Entry</th><th>Exit</th><th>PnL</th><th>Close Time</th></tr></thead>
        <tbody id=\"closedRows\"></tbody>
      </table>
    </div>
  </div>

<script>
function pnlClass(v) {
  const n = Number(v || 0);
  return n >= 0 ? 'pnl-pos' : 'pnl-neg';
}

async function refresh() {
  const r = await fetch('/api/state');
  const s = await r.json();

  const open = Object.values(s.open_positions || {});
  const closed = (s.closed_trades || []).slice(-25).reverse();

  document.getElementById('balance').textContent = Number(s.balance || 0).toFixed(2);
  document.getElementById('openCount').textContent = open.length;
  document.getElementById('closedCount').textContent = closed.length;

  document.getElementById('openRows').innerHTML = open.map(p =>
    `<tr><td>${p.symbol}</td><td>${p.direction}</td><td>${Number(p.entry_price||0).toFixed(4)}</td><td>${Number(p.stop_loss||0).toFixed(4)}</td><td>${Number(p.quantity||0).toFixed(4)}</td><td class="${pnlClass(p.pnl)}">${Number(p.pnl||0).toFixed(2)}</td></tr>`
  ).join('');

  document.getElementById('closedRows').innerHTML = closed.map(t =>
    `<tr><td>${t.symbol}</td><td>${t.status}</td><td>${Number(t.entry_price||0).toFixed(4)}</td><td>${Number(t.exit_price||0).toFixed(4)}</td><td class="${pnlClass(t.pnl)}">${Number(t.pnl||0).toFixed(2)}</td><td>${t.close_time||''}</td></tr>`
  ).join('');
}

refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>
"""


def main() -> None:
    host = CONFIG.dashboard.host
    port = CONFIG.dashboard.port
    server = HTTPServer((host, port), DashboardHandler)
    print(f"Dashboard running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
