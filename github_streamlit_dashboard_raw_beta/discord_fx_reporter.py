#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple
from urllib import request

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from symbols import short_symbol
from tv import TradingViewClient, TradingViewError, resolution_from_interval


PAIR_LIST = [
    ("FOREXCOM:EURJPY", "FOREXCOM:CADJPY"),
    ("FOREXCOM:AUDJPY", "FOREXCOM:GBPJPY"),
    ("FOREXCOM:GBPJPY", "FOREXCOM:AUDJPY"),
    ("FOREXCOM:EURUSD", "FOREXCOM:USDCAD"),
    ("FOREXCOM:GBPUSD", "FOREXCOM:USDCAD"),
    ("FOREXCOM:XAGUSD", "FOREXCOM:XAUUSD"),
    ("FOREXCOM:EURUSD", "FOREXCOM:GBPUSD"),
    ("FOREXCOM:GBPJPY", "FOREXCOM:EURJPY"),
    ("FOREXCOM:AUDUSD", "FOREXCOM:GBPUSD"),
    ("FOREXCOM:NZDUSD", "FOREXCOM:EURUSD"),
    ("FOREXCOM:NZDUSD", "FOREXCOM:GBPUSD"),
    ("FOREXCOM:AUDUSD", "FOREXCOM:NZDUSD"),
    ("FOREXCOM:AUDUSD", "FOREXCOM:EURUSD"),
]
BASE_SYMBOLS = list(dict.fromkeys([base for base, _ in PAIR_LIST]))
FX_SYMBOLS = list(dict.fromkeys([base for base, _ in PAIR_LIST] + [hedge for _, hedge in PAIR_LIST]))

FX_CONTRACT_MULTIPLIER = 100_000.0
LOT_STEP = 0.01
CONTRACT_SIZES = {
    "XAUUSD": 1.0,
    "XAGUSD": 50.0,
}


def filter_weekends(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df.index.dayofweek < 5]


def contract_size(symbol: str) -> float:
    letters = "".join(ch for ch in (symbol or "") if ch.isalpha()).upper()
    core = letters[:6] if len(letters) >= 6 else letters
    return float(CONTRACT_SIZES.get(core, FX_CONTRACT_MULTIPLIER))


def fetch_ohlcv(symbol: str, interval: str, bars: int) -> pd.DataFrame:
    client = TradingViewClient()
    resolution = resolution_from_interval(interval)
    df = client.get_ohlcv(symbol=symbol, resolution=resolution, bars=bars)
    if df.empty:
        return df
    df = df.sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df.tail(int(bars))


def load_ohlcv_map(symbols: List[str], interval: str, bars: int) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    data: Dict[str, pd.DataFrame] = {}
    missing: List[str] = []
    for symbol in symbols:
        error = None
        try:
            df = fetch_ohlcv(symbol, interval, bars)
        except TradingViewError as exc:
            df = pd.DataFrame()
            error = str(exc)
        if df.empty:
            suffix = error or "no data"
            missing.append(f"{symbol} ({suffix})")
        data[symbol] = df
    return data, missing


def clean_ohlcv(df: pd.DataFrame, exclude_weekends: bool) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if exclude_weekends:
        df = filter_weekends(df)
    return df.dropna(subset=["close"])


def build_close_frame(data: Dict[str, pd.DataFrame], labels: Dict[str, str]) -> pd.DataFrame:
    series: Dict[str, pd.Series] = {}
    for symbol, df in data.items():
        if df.empty:
            continue
        label = labels[symbol]
        series[label] = df["close"].copy()
    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series).sort_index()


def compute_returns(close: pd.DataFrame) -> pd.DataFrame:
    if close.empty:
        return close
    returns = np.log(close / close.shift(1))
    return returns.replace([np.inf, -np.inf], np.nan)


def kalman_filter_beta(
    y: pd.Series,
    x: pd.Series,
    process_var: float,
    obs_var: float,
) -> pd.DataFrame:
    data = pd.concat([y, x], axis=1).dropna()
    if data.empty:
        return pd.DataFrame(columns=["alpha", "beta", "resid"])
    data = data.sort_index()

    yv = data.iloc[:, 0].to_numpy(dtype=float)
    xv = data.iloc[:, 1].to_numpy(dtype=float)
    n = len(data)

    process_var = max(float(process_var), 1e-12)
    obs_var = max(float(obs_var), 1e-12)

    state = np.zeros(2)
    p_mat = np.eye(2) * 10.0
    q_mat = np.eye(2) * process_var
    i_mat = np.eye(2)

    alpha = np.zeros(n)
    beta = np.zeros(n)
    resid = np.zeros(n)

    for i in range(n):
        xi = xv[i]
        yi = yv[i]

        p_mat = p_mat + q_mat
        h_vec = np.array([1.0, xi])
        s_val = float(h_vec @ p_mat @ h_vec.T + obs_var)
        if not np.isfinite(s_val) or s_val <= 0:
            s_val = obs_var
        k_vec = (p_mat @ h_vec) / s_val
        y_pred = float(h_vec @ state)
        innov = yi - y_pred
        state = state + k_vec * innov
        p_mat = (i_mat - np.outer(k_vec, h_vec)) @ p_mat

        alpha[i] = state[0]
        beta[i] = state[1]
        resid[i] = yi - (state[0] + state[1] * xi)

    return pd.DataFrame({"alpha": alpha, "beta": beta, "resid": resid}, index=data.index)


def zscore(series: pd.Series) -> pd.Series:
    series = series.dropna()
    if series.empty:
        return pd.Series(dtype=float)
    std = float(series.std())
    if not np.isfinite(std) or std == 0:
        return pd.Series(dtype=float, index=series.index)
    return (series - series.mean()) / std


def latest_value(series: pd.Series) -> float:
    series = series.dropna()
    if series.empty:
        return float("nan")
    return float(series.iloc[-1])


def split_pair(symbol: str) -> Tuple[str, str]:
    s = (symbol or "").strip().upper()
    if len(s) == 6:
        return s[:3], s[3:]
    return s, ""


def units_from_usd(symbol: str, usd_value: float, price: float) -> float:
    if not np.isfinite(price) or price <= 0:
        return float("nan")
    base_ccy, quote_ccy = split_pair(symbol)
    if quote_ccy == "USD":
        return usd_value / price
    if base_ccy == "USD":
        return usd_value
    return usd_value / price


def snap_beta_to_lot_steps(beta: float, base_lot: float, lot_step: float) -> float:
    if not np.isfinite(beta):
        return beta
    if lot_step <= 0 or base_lot <= 0:
        return beta
    base_steps = int(round(base_lot / lot_step))
    if base_steps <= 0:
        return beta
    ratio_steps = int(round(abs(beta) * base_steps))
    if ratio_steps == 0:
        return 0.0
    return float(np.sign(beta) * (ratio_steps / base_steps))


def signal_from_zscore(z_val: float) -> str:
    if not np.isfinite(z_val):
        return "N/A"
    abs_z = abs(z_val)
    if abs_z >= 3.0:
        return "EXTREME"
    if abs_z >= 2.0:
        return "STRONG"
    if abs_z >= 1.0:
        return "MODERATE"
    return "NEUTRAL"


def base_side_from_zscore(z_val: float) -> str:
    if not np.isfinite(z_val):
        return "N/A"
    if z_val > 0:
        return "SELL"
    if z_val < 0:
        return "BUY"
    return "HOLD"


def format_ratio(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    text = f"{value:.3f}"
    return text.rstrip("0").rstrip(".")


def format_size(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.2f}"


def build_pair_compare(
    pair_labels: List[Tuple[str, str]],
    ret: pd.DataFrame,
    close: pd.DataFrame,
    process_var: float,
    obs_var: float,
    base_lot: float,
    use_lcm_size: bool,
) -> pd.DataFrame:
    rows: List[Dict[str, float | str]] = []

    for base, hedge in pair_labels:
        if base not in ret.columns or hedge not in ret.columns:
            continue
        pair = ret[[base, hedge]].dropna()
        if len(pair) < 5:
            continue
        kf = kalman_filter_beta(pair[base], pair[hedge], process_var, obs_var)
        if kf.empty:
            continue
        beta_series = kf["beta"]
        resid_z = zscore(kf["resid"])

        last_beta = latest_value(beta_series)
        beta_used = snap_beta_to_lot_steps(last_beta, base_lot, LOT_STEP) if use_lcm_size else last_beta
        z_val = latest_value(resid_z)
        corr = float(pair[base].corr(pair[hedge])) if len(pair) >= 2 else float("nan")

        base_side_label = base_side_from_zscore(z_val)
        if base_side_label == "SELL":
            base_sign = -1.0
        elif base_side_label == "BUY":
            base_sign = 1.0
        else:
            base_sign = 0.0

        base_last = latest_value(close[base]) if base in close.columns else float("nan")
        base_units = float(base_lot) * contract_size(base)
        base_ccy, quote_ccy = split_pair(base)
        if np.isfinite(base_last):
            if quote_ccy == "USD":
                base_usd = base_last * base_units
            elif base_ccy == "USD":
                base_usd = base_units
            else:
                base_usd = base_last * base_units
        else:
            base_usd = float("nan")
        base_usd_signed = base_usd * base_sign if np.isfinite(base_usd) else float("nan")

        hedge_last = latest_value(close[hedge]) if hedge in close.columns else float("nan")
        if np.isfinite(beta_used) and np.isfinite(base_usd_signed):
            hedge_usd = -beta_used * base_usd_signed
        else:
            hedge_usd = float("nan")
        if not np.isfinite(hedge_usd):
            hedge_side_label = "N/A"
        elif hedge_usd == 0:
            hedge_side_label = "HOLD"
        else:
            hedge_side_label = "SELL" if hedge_usd < 0 else "BUY"

        hedge_units = units_from_usd(hedge, abs(hedge_usd), hedge_last)
        hedge_lots = hedge_units / contract_size(hedge) if np.isfinite(hedge_units) else float("nan")
        if np.isfinite(hedge_lots):
            hedge_lots = float(np.round(hedge_lots / LOT_STEP) * LOT_STEP)

        rows.append(
            {
                "pair": f"{base}/{hedge}",
                "base": base,
                "hedge": hedge,
                "z_score": z_val,
                "hedging_ratio": abs(beta_used) if np.isfinite(beta_used) else float("nan"),
                "beta_used": beta_used,
                "corr": corr,
                "base_lot": base_lot,
                "hedge_lot": hedge_lots,
                "base_side": base_side_label,
                "hedge_side": hedge_side_label,
                "signal": signal_from_zscore(z_val),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("z_score", key=lambda s: s.abs(), ascending=False)
    return df


def plot_pair_compare(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    color_vals = df["hedging_ratio"].fillna(0.0)
    custom = np.stack([df["corr"].fillna(0.0)], axis=-1)
    fig.add_trace(
        go.Scatter(
            x=df["z_score"],
            y=df["hedging_ratio"],
            mode="markers+text",
            text=df["pair"],
            textposition="top center",
            marker=dict(
                size=14,
                color=color_vals,
                colorscale="YlOrRd",
                showscale=True,
                colorbar=dict(title="Hedging Ratio"),
                line=dict(width=1, color="#1f2933"),
            ),
            customdata=custom,
            hovertemplate=(
                "Pair: %{text}<br>"
                "Z-Score: %{x:.3f}<br>"
                "Hedging Ratio: %{y:.3f}<br>"
                "Corr: %{customdata[0]:.3f}<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=-2, line_color="#16a34a", line_dash="dash")
    fig.add_vline(x=0, line_color="#94a3b8", line_dash="dot")
    fig.add_vline(x=2, line_color="#dc2626", line_dash="dash")
    fig.update_layout(
        title="Z-Score vs Hedging Ratio (All Pairs)",
        xaxis_title="Z-Score",
        yaxis_title="Hedging Ratio",
        height=520,
        font=dict(color="#DDE7F4", family="IBM Plex Sans, Segoe UI, Tahoma, sans-serif", size=12),
        paper_bgcolor="#0B1220",
        plot_bgcolor="#0C2340",
        xaxis=dict(gridcolor="#1E2A45", zerolinecolor="#1E2A45"),
        yaxis=dict(gridcolor="#1E2A45", zerolinecolor="#1E2A45"),
    )
    return fig


def build_report_text(df: pd.DataFrame) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "Statistical Arbitrage Update",
        "Statistical Arbitrage Report",
        f"Generated: {generated}",
        "",
        "Report",
        "",
    ]
    for idx, row in enumerate(df.itertuples(index=False), start=1):
        z_val = float(row.z_score) if np.isfinite(row.z_score) else float("nan")
        hedge_ratio = float(row.hedging_ratio) if np.isfinite(row.hedging_ratio) else float("nan")
        base_size = float(row.base_lot) if np.isfinite(row.base_lot) else float("nan")
        hedge_size = float(row.hedge_lot) if np.isfinite(row.hedge_lot) else float("nan")
        lines.extend(
            [
                f"{idx}. {row.pair}",
                f"Z-Score: {z_val:+.3f}" if np.isfinite(z_val) else "Z-Score: n/a",
                f"hedging_ratio : {format_ratio(hedge_ratio)}",
                f"Base : {row.base_side}, Hedge : {row.hedge_side}",
                f"base_size : {format_size(base_size)}",
                f"hedging_size : {format_size(hedge_size)}",
                "",
                f"Signal: {signal_from_zscore(z_val)}",
                "",
            ]
        )
    return "\n".join(lines).rstrip()


def build_chart_png(fig: go.Figure) -> bytes | None:
    try:
        return pio.to_image(fig, format="png", width=1200, height=700, scale=2)
    except Exception as exc:
        print(f"[WARN] Failed to render chart PNG: {exc}")
        return None


def post_webhook(webhook_url: str, content: str, image_bytes: bytes | None) -> None:
    payload = {"content": content}
    if image_bytes is None:
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            webhook_url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "fx-hedge-reporter/1.0",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=30) as resp:
                resp.read()
        except request.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            print(f"[ERROR] Webhook failed: {exc.code} {exc.reason}")
            if detail:
                print(detail)
            raise
        return

    boundary = "----fx-hedge-boundary"
    payload_json = json.dumps(payload)
    lines: List[bytes] = []
    lines.append(f"--{boundary}\r\n".encode("utf-8"))
    lines.append(b'Content-Disposition: form-data; name="payload_json"\r\n\r\n')
    lines.append(payload_json.encode("utf-8"))
    lines.append(b"\r\n")
    lines.append(f"--{boundary}\r\n".encode("utf-8"))
    lines.append(
        b'Content-Disposition: form-data; name="files[0]"; filename="zscore_vs_hedging_ratio.png"\r\n'
    )
    lines.append(b"Content-Type: image/png\r\n\r\n")
    lines.append(image_bytes)
    lines.append(b"\r\n")
    lines.append(f"--{boundary}--\r\n".encode("utf-8"))
    body = b"".join(lines)
    req = request.Request(
        webhook_url,
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "fx-hedge-reporter/1.0",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=60) as resp:
            resp.read()
    except request.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"[ERROR] Webhook failed: {exc.code} {exc.reason}")
        if detail:
            print(detail)
        raise


def load_webhook_url() -> str:
    env_url = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
    if env_url:
        return env_url
    path = Path(__file__).with_name("discord_webhook.txt")
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


def run_once(args: argparse.Namespace) -> None:
    labels = {symbol: short_symbol(symbol) for symbol in FX_SYMBOLS}
    pair_labels: List[Tuple[str, str]] = []
    for base_sym, hedge_sym in PAIR_LIST:
        base_label = labels.get(base_sym)
        hedge_label = labels.get(hedge_sym)
        if base_label and hedge_label:
            pair_labels.append((base_label, hedge_label))

    data, missing = load_ohlcv_map(FX_SYMBOLS, args.interval, args.bars)
    if missing:
        print(f"[WARN] Missing data: {', '.join(missing)}")

    cleaned = {
        symbol: clean_ohlcv(df, exclude_weekends=not args.include_weekends)
        for symbol, df in data.items()
    }
    close = build_close_frame(cleaned, labels)
    if close.empty:
        print("[WARN] No data returned from TradingView.")
        return

    ret = compute_returns(close)
    ret = ret.dropna(how="all")
    if ret.empty:
        print("[WARN] Not enough data to compute returns.")
        return

    pair_compare = build_pair_compare(
        pair_labels=pair_labels,
        ret=ret,
        close=close,
        process_var=args.kalman_q,
        obs_var=args.kalman_r,
        base_lot=args.base_lot,
        use_lcm_size=False,
    )
    if pair_compare.empty:
        print("[WARN] Pair compare data is empty.")
        return

    report_text = build_report_text(pair_compare)
    fig = plot_pair_compare(pair_compare)
    image_bytes = build_chart_png(fig)
    post_webhook(args.webhook_url, report_text, image_bytes)
    print(f"[INFO] Sent report at {datetime.now(timezone.utc).isoformat()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send FX hedge report to Discord on a schedule.")
    parser.add_argument("--webhook-url", default=load_webhook_url())
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--bars", type=int, default=252)
    parser.add_argument("--base-lot", type=float, default=1.0)
    parser.add_argument("--kalman-q", type=float, default=1e-5)
    parser.add_argument("--kalman-r", type=float, default=1e-3)
    parser.add_argument("--include-weekends", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=3600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.webhook_url:
        raise SystemExit("Missing webhook URL. Set DISCORD_WEBHOOK_URL or pass --webhook-url.")

    run_once(args)
    while True:
        time.sleep(max(args.interval_seconds, 60))
        run_once(args)


if __name__ == "__main__":
    main()
