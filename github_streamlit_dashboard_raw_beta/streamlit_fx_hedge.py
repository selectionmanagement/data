#!/usr/bin/env python
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from tv import TradingViewClient, TradingViewError, resolution_from_interval
from symbols import short_symbol

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
DEFAULT_INTERVAL = "1h"
DEFAULT_BARS = 720
DEFAULT_BUFFER_BARS = 72
DEFAULT_MIN_SEGMENT_BARS = 24
DEFAULT_KALMAN_PROCESS_VAR = 1e-5
DEFAULT_KALMAN_OBS_VAR = 1e-3
FX_CONTRACT_MULTIPLIER = 100_000.0
LOT_STEP = 0.01
CACHE_TTL_SECONDS = 30 * 60
CONTRACT_SIZES = {
    "XAUUSD": 1.0,
    "XAGUSD": 50.0,
}
VOL_TARGET_WINDOWS = {
    "15m": 96,
    "30m": 48,
    "60m": 24,
    "1h": 24,
}
VOL_TARGET_MIN_SCALE = 0.5
VOL_TARGET_MAX_SCALE = 1.5

PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font={"family": "IBM Plex Sans, Segoe UI, Tahoma, sans-serif", "color": "#DDE7F4", "size": 12},
        paper_bgcolor="#0B1220",
        plot_bgcolor="#0C2340",
        colorway=["#38BDF8", "#0EA5E9", "#22D3EE", "#14B8A6", "#F59E0B", "#F97316", "#F43F5E"],
        xaxis={"gridcolor": "#1E2A45", "zerolinecolor": "#1E2A45"},
        yaxis={"gridcolor": "#1E2A45", "zerolinecolor": "#1E2A45"},
    )
)
pio.templates["fx_light"] = PLOTLY_TEMPLATE
pio.templates.default = "fx_light"

FX_CSS = """
<style>
:root {
    --fx-bg: #0B1220;
    --fx-bg-2: #0F1E37;
    --fx-panel: #0C2340;
    --fx-panel-2: #0A315C;
    --fx-text: #DDE7F4;
    --fx-muted: #9BB2D1;
    --fx-accent: #38BDF8;
    --fx-accent-2: #0EA5E9;
    --fx-border: #1E2A45;
    --fx-input-bg: #0B1220;
    --fx-input-text: #E6EEF8;
}
html, body, [data-testid="stAppViewContainer"] {
    background: radial-gradient(900px 700px at 10% -10%, #0A315C 0%, transparent 60%),
                radial-gradient(900px 700px at 110% 10%, #0F1E37 0%, transparent 55%),
                linear-gradient(180deg, #0F1E37 0%, #0C2340 55%, #0B1220 100%);
    color: var(--fx-text);
    font-family: "IBM Plex Sans", "Segoe UI", "Tahoma", sans-serif;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0C2340 0%, #0B1220 100%);
    border-right: 1px solid var(--fx-border);
    color: var(--fx-input-text);
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 {
    color: var(--fx-accent);
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: var(--fx-input-text);
}
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea {
    background-color: var(--fx-input-bg) !important;
    color: var(--fx-input-text) !important;
    border: 1px solid var(--fx-border) !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: var(--fx-input-bg) !important;
    color: var(--fx-input-text) !important;
    border: 1px solid var(--fx-border) !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] span {
    color: var(--fx-input-text) !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] svg {
    color: var(--fx-input-text) !important;
}
.block-container {
    padding-top: 1.5rem;
    animation: rise 0.5s ease-out;
}
@keyframes rise {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
.stTabs [data-baseweb="tab"] {
    background: var(--fx-panel);
    border-radius: 999px;
    border: 1px solid var(--fx-border);
    margin-right: 0.35rem;
    padding: 0.25rem 0.8rem;
    color: var(--fx-text);
}
.stTabs [aria-selected="true"] {
    color: var(--fx-accent);
    border-color: var(--fx-accent);
    box-shadow: 0 6px 18px rgba(14, 165, 233, 0.25);
}
.stButton > button {
    background: var(--fx-accent-2);
    border: 1px solid var(--fx-accent-2);
    color: #06121f;
    font-weight: 600;
}
.stButton > button:hover {
    background: var(--fx-accent);
    border-color: var(--fx-accent);
}
div[data-testid="stMetric"] {
    background: linear-gradient(180deg, #0A315C 0%, #0C2340 100%);
    border: 1px solid var(--fx-border);
    border-radius: 12px;
    padding: 0.4rem 0.6rem;
}
div[data-testid="stPlotlyChart"] {
    background: var(--fx-panel);
    border-radius: 12px;
    padding: 0.4rem;
}
div[data-testid="stDataFrame"] {
    background: var(--fx-panel);
    border: 1px solid var(--fx-border);
    border-radius: 12px;
}
</style>
"""


def filter_weekends(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df.index.dayofweek < 5]


def contract_size(symbol: str) -> float:
    letters = "".join(ch for ch in (symbol or "") if ch.isalpha()).upper()
    core = letters[:6] if len(letters) >= 6 else letters
    return float(CONTRACT_SIZES.get(core, FX_CONTRACT_MULTIPLIER))


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def fetch_ohlcv(symbol: str, interval: str, bars: int) -> pd.DataFrame:
    client = TradingViewClient()
    resolution = resolution_from_interval(interval)
    df = client.get_ohlcv(symbol=symbol, resolution=resolution, bars=bars)
    if df.empty:
        return df
    df = df.sort_index()
    return df.tail(int(bars))


def load_ohlcv_map(
    symbols: List[str],
    interval: str,
    bars: int,
) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    data: Dict[str, pd.DataFrame] = {}
    missing: List[str] = []
    progress = st.progress(0.0)
    total = max(len(symbols), 1)

    for idx, symbol in enumerate(symbols, start=1):
        error: str | None = None
        try:
            df = fetch_ohlcv(symbol, interval, bars)
        except TradingViewError as exc:
            df = pd.DataFrame()
            error = str(exc)
        if df.empty:
            suffix = error or "no data"
            missing.append(f"{symbol} ({suffix})")
        data[symbol] = df
        progress.progress(min(idx / total, 1.0))

    progress.empty()
    return data, missing


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def fetch_ohlcv_range(
    symbol: str,
    interval: str,
    start_iso: str,
    end_iso: str,
    bars_per_request: int = 5000,
) -> pd.DataFrame:
    client = TradingViewClient()
    resolution = resolution_from_interval(interval)
    start_ts = pd.Timestamp(start_iso) if start_iso else None
    end_ts = pd.Timestamp(end_iso) if end_iso else None
    df = client.get_ohlcv(
        symbol=symbol,
        resolution=resolution,
        bars=int(bars_per_request),
        start=start_ts,
        end=end_ts,
        max_bars_per_request=int(bars_per_request),
    )
    if df.empty:
        return df
    return df.sort_index()


def load_ohlcv_range_map(
    symbols: List[str],
    interval: str,
    start_iso: str,
    end_iso: str,
    bars_per_request: int = 5000,
) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    data: Dict[str, pd.DataFrame] = {}
    missing: List[str] = []
    progress = st.progress(0.0)
    total = max(len(symbols), 1)

    for idx, symbol in enumerate(symbols, start=1):
        error: str | None = None
        try:
            df = fetch_ohlcv_range(symbol, interval, start_iso, end_iso, bars_per_request)
        except TradingViewError as exc:
            df = pd.DataFrame()
            error = str(exc)
        if df.empty:
            suffix = error or "no data"
            missing.append(f"{symbol} ({suffix})")
        data[symbol] = df
        progress.progress(min(idx / total, 1.0))

    progress.empty()
    return data, missing


def split_frame_on_gaps(df: pd.DataFrame, gap_hours: float) -> List[pd.DataFrame]:
    if df.empty:
        return []
    gap_seconds = float(gap_hours) * 3600.0
    if not np.isfinite(gap_seconds) or gap_seconds <= 0:
        return [df]
    df = df.sort_index()
    gaps = df.index.to_series().diff().dt.total_seconds() > gap_seconds
    if not gaps.any():
        return [df]
    group_ids = gaps.cumsum()
    return [segment for _, segment in df.groupby(group_ids)]


def clean_ohlcv(
    df: pd.DataFrame,
    *,
    exclude_weekends: bool,
    gap_hours: float | None,
    gap_action: str,
    min_segment_bars: int,
    target_bars: int,
) -> Tuple[pd.DataFrame, Dict[str, float | int]]:
    stats = {
        "raw_bars": 0,
        "deduped_bars": 0,
        "weekend_bars": 0,
        "gap_count": 0,
        "segments_kept": 0,
        "final_bars": 0,
        "dropped_bars": 0,
    }
    if df.empty:
        return df, stats

    df = df.sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    stats["raw_bars"] = len(df)
    df = df[~df.index.duplicated(keep="last")]
    stats["deduped_bars"] = len(df)

    if exclude_weekends:
        df = filter_weekends(df)
    stats["weekend_bars"] = len(df)

    gap_count = 0
    if gap_hours is not None:
        gap_seconds = float(gap_hours) * 3600.0
        if np.isfinite(gap_seconds) and gap_seconds > 0:
            gaps = df.index.to_series().diff().dt.total_seconds() > gap_seconds
            gap_count = int(gaps.sum())
            if gap_action == "drop":
                df = df.loc[~gaps]
    stats["gap_count"] = gap_count

    segments_kept = 0
    if gap_hours is not None and min_segment_bars > 1:
        segments = split_frame_on_gaps(df, gap_hours)
        kept = [segment for segment in segments if len(segment) >= min_segment_bars]
        segments_kept = len(kept)
        df = pd.concat(kept) if kept else df.iloc[0:0]
    else:
        segments = split_frame_on_gaps(df, gap_hours) if gap_hours is not None else [df]
        segments_kept = len(segments)
    stats["segments_kept"] = segments_kept

    if target_bars > 0:
        df = df.tail(int(target_bars))
    stats["final_bars"] = len(df)
    stats["dropped_bars"] = max(int(stats["raw_bars"] - stats["final_bars"]), 0)

    return df, stats


def clean_ohlcv_map(
    data: Dict[str, pd.DataFrame],
    labels: Dict[str, str],
    *,
    exclude_weekends: bool,
    gap_hours: float | None,
    gap_action: str,
    min_segment_bars: int,
    target_bars: int,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    cleaned: Dict[str, pd.DataFrame] = {}
    rows: List[Dict[str, float | int | str]] = []
    for symbol, df in data.items():
        label = labels.get(symbol, symbol)
        cleaned_df, stats = clean_ohlcv(
            df,
            exclude_weekends=exclude_weekends,
            gap_hours=gap_hours,
            gap_action=gap_action,
            min_segment_bars=min_segment_bars,
            target_bars=target_bars,
        )
        stats_row = {"symbol": label}
        stats_row.update(stats)
        rows.append(stats_row)
        cleaned[symbol] = cleaned_df

    stats_df = pd.DataFrame(rows)
    return cleaned, stats_df


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


def split_pair(symbol: str) -> Tuple[str, str]:
    s = (symbol or "").strip().upper()
    if len(s) == 6:
        return s[:3], s[3:]
    return s, ""


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


def pip_value_usd(symbol: str, price: float, contract_units: float) -> float:
    if not np.isfinite(price) or price <= 0:
        return float("nan")
    base_ccy, quote_ccy = split_pair(symbol)
    pip = 0.01 if quote_ccy == "JPY" else 0.0001
    if quote_ccy == "USD":
        return pip * contract_units
    if base_ccy == "USD":
        return (pip / price) * contract_units
    return float("nan")


def base_value_usd(symbol: str, price: float, units: float) -> float:
    if not np.isfinite(price) or price <= 0:
        return float("nan")
    base_ccy, quote_ccy = split_pair(symbol)
    if quote_ccy == "USD":
        return price * units
    if base_ccy == "USD":
        return units
    return price * units


def units_from_usd(symbol: str, usd_value: float, price: float) -> float:
    if not np.isfinite(price) or price <= 0:
        return float("nan")
    base_ccy, quote_ccy = split_pair(symbol)
    if quote_ccy == "USD":
        return usd_value / price
    if base_ccy == "USD":
        return usd_value
    return usd_value / price


def pnl_usd(symbol: str, units: float, prev_price: float, price: float) -> float:
    if not np.isfinite(prev_price) or not np.isfinite(price) or prev_price <= 0 or price <= 0:
        return 0.0
    base_ccy, quote_ccy = split_pair(symbol)
    delta = price - prev_price
    if quote_ccy == "USD":
        return units * delta
    if base_ccy == "USD":
        return units * delta / price
    return units * delta


def normalize_close(close: pd.DataFrame, basis: str) -> pd.DataFrame:
    if close.empty:
        return close
    if basis == "anchor":
        first = close.apply(lambda s: s.dropna().iloc[0] if not s.dropna().empty else np.nan)
        first = first.replace(0, np.nan)
        return close.divide(first)
    if basis == "reset daily":
        daily_anchor = close.groupby(close.index.floor("D")).transform("first")
        daily_anchor = daily_anchor.replace(0, np.nan)
        return close.divide(daily_anchor)
    return close


def drop_gap_rows(data: pd.DataFrame, gap_hours: float) -> pd.DataFrame:
    if data.empty:
        return data
    gap_seconds = float(gap_hours) * 3600.0
    if not np.isfinite(gap_seconds) or gap_seconds <= 0:
        return data
    data = data.sort_index()
    gaps = data.index.to_series().diff().dt.total_seconds()
    drop_mask = gaps > gap_seconds
    if drop_mask.any():
        return data.loc[~drop_mask]
    return data


def split_on_gaps(series: pd.Series, gap_hours: float) -> List[pd.Series]:
    if series.empty:
        return []
    gap_seconds = float(gap_hours) * 3600.0
    if not np.isfinite(gap_seconds) or gap_seconds <= 0:
        return [series]
    series = series.sort_index()
    gaps = series.index.to_series().diff().dt.total_seconds() > gap_seconds
    if not gaps.any():
        return [series]
    group_ids = gaps.cumsum()
    return [series[group_ids == gid] for gid in range(int(group_ids.max()) + 1)]


def break_series_gaps(series: pd.Series, gap_hours: float | None) -> pd.Series:
    if series.empty or gap_hours is None:
        return series
    gap_seconds = float(gap_hours) * 3600.0
    if not np.isfinite(gap_seconds) or gap_seconds <= 0:
        return series
    series = series.sort_index()
    new_index: List[pd.Timestamp] = []
    new_values: List[float] = []
    prev_ts: pd.Timestamp | None = None
    for ts, val in series.items():
        if prev_ts is not None and (ts - prev_ts).total_seconds() > gap_seconds:
            new_index.append(prev_ts + pd.Timedelta(seconds=1))
            new_values.append(float("nan"))
        new_index.append(ts)
        new_values.append(float(val) if np.isfinite(val) else float("nan"))
        prev_ts = ts
    return pd.Series(new_values, index=pd.DatetimeIndex(new_index))


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
    gap_hours: float | None = None,
    gap_action: str = "none",
) -> pd.DataFrame:
    data = pd.concat([y, x], axis=1).dropna()
    if data.empty:
        return pd.DataFrame(columns=["alpha", "beta", "resid"])
    data = data.sort_index()
    if gap_action == "drop" and gap_hours is not None:
        data = drop_gap_rows(data, gap_hours)
        if data.empty:
            return pd.DataFrame(columns=["alpha", "beta", "resid"])

    yv = data.iloc[:, 0].to_numpy(dtype=float)
    xv = data.iloc[:, 1].to_numpy(dtype=float)
    n = len(data)
    idx = data.index

    process_var = max(float(process_var), 1e-12)
    obs_var = max(float(obs_var), 1e-12)
    gap_seconds = None
    if gap_action == "reset" and gap_hours is not None:
        gap_seconds = max(float(gap_hours) * 3600.0, 0.0)

    state = np.zeros(2)
    p_mat = np.eye(2) * 10.0
    q_mat = np.eye(2) * process_var
    i_mat = np.eye(2)

    alpha = np.zeros(n)
    beta = np.zeros(n)
    resid = np.zeros(n)

    prev_ts = None
    for i in range(n):
        if gap_seconds is not None and prev_ts is not None:
            delta = (idx[i] - prev_ts).total_seconds()
            if delta > gap_seconds:
                state = np.zeros(2)
                p_mat = np.eye(2) * 10.0
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
        prev_ts = idx[i]

    return pd.DataFrame({"alpha": alpha, "beta": beta, "resid": resid}, index=data.index)


def zscore(series: pd.Series, gap_hours: float | None, gap_action: str) -> pd.Series:
    series = series.dropna()
    if series.empty:
        return pd.Series(dtype=float)
    if gap_action != "none" and gap_hours is not None:
        segments = split_on_gaps(series, gap_hours)
        scored: List[pd.Series] = []
        for segment in segments:
            std = float(segment.std())
            if not np.isfinite(std) or std == 0:
                scored.append(pd.Series(dtype=float, index=segment.index))
            else:
                scored.append((segment - segment.mean()) / std)
        if not scored:
            return pd.Series(dtype=float)
        return pd.concat(scored).sort_index()
    std = float(series.std())
    if not np.isfinite(std) or std == 0:
        return pd.Series(dtype=float, index=series.index)
    return (series - series.mean()) / std


def latest_value(series: pd.Series) -> float:
    series = series.dropna()
    if series.empty:
        return float("nan")
    return float(series.iloc[-1])


def vol_window_from_interval(interval: str) -> int:
    if not interval:
        return VOL_TARGET_WINDOWS["1h"]
    return int(VOL_TARGET_WINDOWS.get(interval.strip().lower(), VOL_TARGET_WINDOWS["1h"]))


def vol_target_scale(resid: pd.Series, interval: str) -> Tuple[float, float, float]:
    resid = resid.dropna()
    window = vol_window_from_interval(interval)
    if resid.empty or window <= 1 or len(resid) < window:
        return 1.0, float("nan"), float("nan")

    rolling = resid.rolling(window).std().dropna()
    if rolling.empty:
        return 1.0, float("nan"), float("nan")

    target_vol = float(rolling.median())
    current_vol = float(rolling.iloc[-1])
    if not np.isfinite(target_vol) or target_vol <= 0 or not np.isfinite(current_vol) or current_vol <= 0:
        return 1.0, current_vol, target_vol

    scale = target_vol / current_vol
    scale = min(max(scale, VOL_TARGET_MIN_SCALE), VOL_TARGET_MAX_SCALE)
    return scale, current_vol, target_vol


def format_ratio(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    text = f"{value:.3f}"
    return text.rstrip("0").rstrip(".")


def format_size(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.2f}"


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


def build_pair_compare(
    pair_labels: List[Tuple[str, str]],
    ret: pd.DataFrame,
    close: pd.DataFrame,
    process_var: float,
    obs_var: float,
    base_lot: float,
    base_side: str,
    interval: str,
    gap_hours: float | None,
    gap_action: str,
    use_lcm_size: bool,
) -> pd.DataFrame:
    rows: List[Dict[str, float | str]] = []
    kalman_gap_action = "reset" if gap_action == "reset" else "none"

    for base, hedge in pair_labels:
        if base not in ret.columns or hedge not in ret.columns:
            continue
        pair = ret[[base, hedge]].dropna()
        if len(pair) < 5:
            continue
        kf = kalman_filter_beta(
            pair[base],
            pair[hedge],
            process_var,
            obs_var,
            gap_hours=gap_hours,
            gap_action=kalman_gap_action,
        )
        if kf.empty:
            continue
        beta_series = kf["beta"]
        resid = kf["resid"].dropna()
        resid_z = zscore(resid, None, "none")

        scale, _, _ = vol_target_scale(resid, interval)
        base_lot_used = base_lot * scale

        last_beta = latest_value(beta_series)
        beta_used = snap_beta_to_lot_steps(last_beta, base_lot_used, LOT_STEP) if use_lcm_size else last_beta
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
        base_units = float(base_lot_used) * contract_size(base)
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
                "base_lot": base_lot_used,
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
        height=420,
        paper_bgcolor="#0B1220",
        plot_bgcolor="#0C2340",
    )
    return fig


def build_pair_report_text(df: pd.DataFrame) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "Statistical Arbitrage Update",
        "📊 Statistical Arbitrage Report",
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


def plot_series(
    series_map: Dict[str, pd.Series],
    title: str,
    y_title: str,
    hlines: List[float] | None = None,
    gap_hours: float | None = None,
) -> go.Figure:
    fig = go.Figure()
    for name, series in series_map.items():
        if series.empty:
            continue
        plot_series_data = break_series_gaps(series, gap_hours)
        fig.add_trace(
            go.Scatter(
                x=plot_series_data.index,
                y=plot_series_data.values,
                name=name,
                mode="lines",
            )
        )
    for y in hlines or []:
        fig.add_hline(y=y, line_color="#9ca3af", line_dash="dash")
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title=y_title,
        height=360,
        paper_bgcolor="#0B1220",
        plot_bgcolor="#0C2340",
    )
    return fig


def build_summary(
    base: str,
    ret: pd.DataFrame,
    close: pd.DataFrame,
    process_var: float,
    obs_var: float,
    base_lot: float,
    base_side: str,
    interval: str,
    gap_hours: float | None,
    gap_action: str,
    use_lcm_size: bool,
    allowed_hedges: List[str] | None,
) -> Tuple[pd.DataFrame, Dict[str, pd.Series], Dict[str, pd.Series]]:
    if base not in ret.columns:
        return pd.DataFrame(), {}, {}

    summary_rows: List[Dict[str, float | str]] = []
    beta_map: Dict[str, pd.Series] = {}
    resid_map: Dict[str, pd.Series] = {}

    base_last = latest_value(close[base]) if base in close.columns else float("nan")
    base_ccy, quote_ccy = split_pair(base)
    kalman_gap_action = "reset" if gap_action == "reset" else "none"

    for other in ret.columns:
        if other == base:
            continue
        if allowed_hedges is not None and other not in allowed_hedges:
            continue
        pair = ret[[base, other]].dropna()
        if len(pair) < 5:
            continue
        kf = kalman_filter_beta(
            pair[base],
            pair[other],
            process_var,
            obs_var,
            gap_hours=gap_hours,
            gap_action=kalman_gap_action,
        )
        if kf.empty:
            continue
        beta_series = kf["beta"]
        beta_map[other] = beta_series
        resid = kf["resid"].dropna()
        resid_z = zscore(resid, None, "none")
        resid_map[other] = resid_z

        scale, _, _ = vol_target_scale(resid, interval)
        base_lot_used = base_lot * scale
        last_beta = latest_value(beta_series)
        beta_used = snap_beta_to_lot_steps(last_beta, base_lot_used, LOT_STEP) if use_lcm_size else last_beta
        last_z = latest_value(resid_z)
        base_side_label = base_side_from_zscore(last_z)
        if base_side_label == "SELL":
            base_sign = -1.0
        elif base_side_label == "BUY":
            base_sign = 1.0
        else:
            base_sign = 0.0
        base_units = float(base_lot_used) * contract_size(base)
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
        corr = float(pair[base].corr(pair[other])) if len(pair) >= 2 else float("nan")
        hedge_last = latest_value(close[other]) if other in close.columns else float("nan")
        hedge_usd = -beta_used * base_usd_signed if np.isfinite(beta_used) else float("nan")
        if not np.isfinite(hedge_usd):
            action = "n/a"
        elif hedge_usd == 0:
            action = "hold"
        else:
            action = "short" if hedge_usd < 0 else "long"

        summary_rows.append(
            {
                "hedge": other,
                "last_beta": last_beta,
                "beta_used": beta_used,
                "corr": corr,
                "resid_z": last_z,
                "base_lot": base_lot_used,
                "base_units": base_units,
                "base_last": base_last,
                "base_usd": base_usd,
                "base_side": base_side_label,
                "base_usd_signed": base_usd_signed,
                "hedge_last": hedge_last,
                "hedge_usd": hedge_usd,
                "action": action,
            }
        )

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values("hedge")
    return summary, beta_map, resid_map


def render_tab(
    base: str,
    close: pd.DataFrame,
    ret: pd.DataFrame,
    process_var: float,
    obs_var: float,
    base_lot: float,
    base_side: str,
    interval: str,
    gap_hours: float | None,
    gap_action: str,
    use_lcm_size: bool,
    allowed_hedges: List[str] | None,
) -> None:
    if base not in close.columns:
        st.error(f"No data for {base}.")
        return

    base_series = close[base].dropna()
    if base_series.empty:
        st.error(f"No data for {base}.")
        return

    start_ts = base_series.index.min()
    end_ts = base_series.index.max()
    st.caption(f"Bars: {len(base_series)} | Range: {start_ts} -> {end_ts}")

    summary, beta_map, resid_map = build_summary(
        base,
        ret,
        close,
        process_var,
        obs_var,
        base_lot,
        base_side,
        interval,
        gap_hours,
        gap_action,
        use_lcm_size,
        allowed_hedges,
    )

    if summary.empty:
        st.info("Not enough data to compute hedge ratios.")
        return

    base_last = float(summary["base_last"].iloc[0]) if "base_last" in summary.columns else float("nan")
    base_lot = float(summary["base_lot"].iloc[0]) if "base_lot" in summary.columns else float("nan")
    base_units = float(summary["base_units"].iloc[0]) if "base_units" in summary.columns else float("nan")
    base_usd = float(summary["base_usd"].iloc[0]) if "base_usd" in summary.columns else float("nan")
    base_side = str(summary["base_side"].iloc[0]) if "base_side" in summary.columns else "BUY"
    base_usd_signed = (
        float(summary["base_usd_signed"].iloc[0]) if "base_usd_signed" in summary.columns else base_usd
    )
    info_cols = st.columns(5)
    info_cols[0].metric("Base price", f"{base_last:.5f}" if np.isfinite(base_last) else "n/a")
    info_cols[1].metric("Base lot", f"{base_lot:.3f}" if np.isfinite(base_lot) else "n/a")
    info_cols[2].metric("Base units", f"{base_units:.2f}" if np.isfinite(base_units) else "n/a")
    info_cols[3].metric("Base side", base_side.upper())
    info_cols[4].metric("Base value (USD)", f"{base_usd_signed:.2f}" if np.isfinite(base_usd_signed) else "n/a")

    trade = summary.copy()
    trade["hedge_usd_abs"] = trade["hedge_usd"].abs()
    trade["base_ccy"] = trade["hedge"].apply(lambda s: split_pair(str(s))[0])
    trade["quote_ccy"] = trade["hedge"].apply(lambda s: split_pair(str(s))[1])
    price_safe = trade["hedge_last"].replace(0, np.nan)
    trade["units"] = np.where(
        trade["quote_ccy"] == "USD",
        trade["hedge_usd_abs"] / price_safe,
        np.where(
            trade["base_ccy"] == "USD",
            trade["hedge_usd_abs"],
            trade["hedge_usd_abs"] / price_safe,
        ),
    )
    trade["contract_size"] = trade["hedge"].apply(contract_size)
    trade["lots"] = trade["units"] / trade["contract_size"]
    trade["lots"] = (trade["lots"] / LOT_STEP).round() * LOT_STEP
    trade["pip_value"] = trade.apply(
        lambda row: pip_value_usd(
            str(row["hedge"]),
            float(row["hedge_last"]),
            contract_size(str(row["hedge"])),
        ),
        axis=1,
    )
    trade["side"] = trade["action"].map({"short": "SELL", "long": "BUY"}).fillna("HOLD")
    trade = trade.sort_values("hedge_usd_abs", ascending=False)

    trade_sheet = trade[
        [
            "hedge",
            "side",
            "lots",
            "units",
            "hedge_usd_abs",
            "pip_value",
            "last_beta",
            "beta_used",
            "resid_z",
            "corr",
        ]
    ].rename(
        columns={
            "hedge_usd_abs": "usd_value",
        }
    )
    for col in ["lots", "units", "usd_value", "pip_value", "last_beta", "beta_used", "resid_z", "corr"]:
        trade_sheet[col] = pd.to_numeric(trade_sheet[col], errors="coerce").round(4)

    st.subheader("Base Position")
    base_contract_size = contract_size(base)
    base_usd_abs = abs(base_usd_signed) if np.isfinite(base_usd_signed) else float("nan")
    base_line = (
        f"(Base: {base}, Value: ${base_usd_abs:.2f}, "
        f"Size: {base_lot:.2f} lot, {base_side.upper()})"
    )
    st.markdown(base_line)

    st.subheader("Trade Instructions")
    instructions = []
    for _, row in trade_sheet.iterrows():
        side = row["side"]
        pair = row["hedge"]
        lots = row["lots"]
        units = row["units"]
        usd_val = row["usd_value"]
        if not np.isfinite(lots) or not np.isfinite(units):
            continue
        instructions.append(
            f"(Pair: {pair}, Value: ${usd_val:.2f}, Size: {lots:.2f} lot, {side})"
        )
    if instructions:
        st.markdown("\n".join(f"- {line}" for line in instructions))
    else:
        st.info("No trade instructions available.")

    st.subheader("Statistical Arbitrage Report")
    if trade_sheet.empty:
        st.info("No report data available.")
    else:
        generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        report_lines = [
            "Statistical Arbitrage Update",
            "📊 Statistical Arbitrage Report",
            f"Generated: {generated}",
            "",
            "Report",
            "",
        ]
        for idx, row in enumerate(trade_sheet.iterrows(), start=1):
            _, data = row
            hedge = str(data.get("hedge", ""))
            pair = f"{base}/{hedge}" if hedge else base
            z_val = float(data["resid_z"]) if np.isfinite(data.get("resid_z", float("nan"))) else float("nan")
            beta_used = float(data["beta_used"]) if np.isfinite(data.get("beta_used", float("nan"))) else float("nan")
            lots = float(data["lots"]) if np.isfinite(data.get("lots", float("nan"))) else float("nan")
            hedge_side = str(data.get("side", "N/A"))
            report_lines.extend(
                [
                    f"{idx}. {pair}",
                    f"Z-Score: {z_val:+.3f}" if np.isfinite(z_val) else "Z-Score: n/a",
                    f"hedging_ratio : {format_ratio(abs(beta_used))}",
                    f"Base : {base_side.upper()}, Hedge : {hedge_side}",
                    f"base_size : {format_size(base_lot)}",
                    f"hedging_size : {format_size(lots)}",
                    "",
                    f"Signal: {signal_from_zscore(z_val)}",
                    "",
                ]
            )
        st.code("\n".join(report_lines).rstrip())

    st.subheader("Trade Sheet")
    st.dataframe(trade_sheet, use_container_width=True)
    st.caption(
        f"Base units = base_lot * {base_contract_size:.0f} | "
        f"usd_value = abs(-beta_used * base_value) | units are base currency; lots rounded to {LOT_STEP:.2f}"
    )

    with st.expander("Diagnostics (full table)", expanded=False):
        display = summary.copy()
        for col in [
            "last_beta",
            "beta_used",
            "corr",
            "resid_z",
            "base_lot",
            "base_units",
            "base_last",
            "base_usd",
            "base_usd_signed",
            "hedge_last",
            "hedge_usd",
        ]:
            display[col] = pd.to_numeric(display[col], errors="coerce").round(4)
        st.dataframe(display, use_container_width=True)

    if beta_map:
        st.subheader("Kalman Hedge Ratio (Beta)")
        fig = plot_series(
            beta_map,
            "Kalman Beta vs Base",
            "beta",
            hlines=[0.0],
            gap_hours=gap_hours,
        )
        st.plotly_chart(fig, use_container_width=True)

    if resid_map:
        st.subheader("Residual Z-Score")
        fig = plot_series(
            resid_map,
            "Residual Z-Score",
            "z",
            hlines=[0.0, 2.0, -2.0],
            gap_hours=gap_hours,
        )
        st.plotly_chart(fig, use_container_width=True)

def main() -> None:
    st.set_page_config(page_title="FX Hedge Ratio Dashboard", layout="wide")
    st.markdown(FX_CSS, unsafe_allow_html=True)
    st.title("FX Hedge Ratio Dashboard")
    st.caption("TradingView FOREXCOM data, cached in memory only.")

    with st.sidebar:
        st.header("Data")
        interval = st.selectbox("Interval", [DEFAULT_INTERVAL], index=0)
        bars = int(
            st.number_input("Bars", min_value=100, max_value=2000, value=DEFAULT_BARS, step=10)
        )
        buffer_bars = int(
            st.number_input(
                "Buffer bars (prefetch)",
                min_value=0,
                max_value=500,
                value=DEFAULT_BUFFER_BARS,
                step=12,
            )
        )
        exclude_weekends = st.checkbox("Exclude weekend bars (UTC)", value=True)
        st.caption(f"Cache TTL: {CACHE_TTL_SECONDS // 60} minutes")
        if st.button("Force refresh"):
            st.cache_data.clear()

        st.header("Cleaning")
        min_segment_bars = int(
            st.number_input(
                "Min segment bars",
                min_value=1,
                max_value=200,
                value=DEFAULT_MIN_SEGMENT_BARS,
                step=1,
            )
        )

        st.header("Hedge model")
        process_var = DEFAULT_KALMAN_PROCESS_VAR
        obs_var = DEFAULT_KALMAN_OBS_VAR
        base_lot = float(
            st.number_input(
                "Base lot size",
                min_value=0.001,
                max_value=100.0,
                value=1.0,
                step=0.01,
                format="%.3f",
            )
        )
        base_side = "AUTO"
        st.caption("Base side: auto (z-score > 0 = SELL, < 0 = BUY)")
        use_lcm_size = True
        st.caption(
            f"1 standard lot = {FX_CONTRACT_MULTIPLIER:.0f} units | "
            "Base value = price * lot * contract units (if quote is USD)"
        )
        st.header("Gap handling")
        st.caption("Gap action: Drop after gap")
        gap_hours = float(
            st.number_input(
                "Gap threshold (hours)",
                min_value=1.0,
                max_value=72.0,
                value=8.0,
                step=1.0,
            )
        )
        gap_action = "drop"

    labels = {symbol: short_symbol(symbol) for symbol in FX_SYMBOLS}
    pair_labels: List[Tuple[str, str]] = []
    base_hedges: Dict[str, List[str]] = {}
    for base_sym, hedge_sym in PAIR_LIST:
        base_label = labels.get(base_sym)
        hedge_label = labels.get(hedge_sym)
        if base_label and hedge_label:
            pair_labels.append((base_label, hedge_label))
            base_hedges.setdefault(base_label, []).append(hedge_label)

    fetch_bars = int(bars + max(buffer_bars, 0))
    with st.spinner("Fetching TradingView data..."):
        data, missing = load_ohlcv_map(FX_SYMBOLS, interval, fetch_bars)

    if missing:
        st.warning("Missing data: " + ", ".join(missing))

    cleaned_data, clean_stats = clean_ohlcv_map(
        data,
        labels,
        exclude_weekends=exclude_weekends,
        gap_hours=gap_hours,
        gap_action=gap_action,
        min_segment_bars=min_segment_bars,
        target_bars=bars,
    )
    cleaned_missing = [labels[symbol] for symbol, df in cleaned_data.items() if df.empty]
    if cleaned_missing:
        st.warning("Empty after cleaning: " + ", ".join(cleaned_missing))
    if not clean_stats.empty:
        clean_stats = clean_stats.sort_values("symbol")
        with st.expander("Data cleaning summary", expanded=False):
            st.dataframe(clean_stats, use_container_width=True)

    close = build_close_frame(cleaned_data, labels)
    if close.empty:
        st.error("No data returned from TradingView.")
        return

    ret = compute_returns(close)
    ret = ret.dropna(how="all")
    if ret.empty:
        st.error("Not enough data to compute returns.")
        return

    pair_compare = build_pair_compare(
        pair_labels,
        ret,
        close,
        process_var,
        obs_var,
        base_lot,
        base_side,
        interval,
        gap_hours,
        gap_action,
        use_lcm_size,
    )

    st.subheader("Z-Score vs Hedging Ratio (All Pairs)")
    if pair_compare.empty:
        st.info("Not enough data to build the comparison chart.")
    else:
        st.plotly_chart(plot_pair_compare(pair_compare), use_container_width=True)
        summary_view = pair_compare[
            [
                "pair",
                "z_score",
                "hedging_ratio",
                "base_lot",
                "hedge_lot",
                "corr",
                "signal",
            ]
        ].copy()
        for col in ["z_score", "hedging_ratio", "base_lot", "hedge_lot", "corr"]:
            summary_view[col] = pd.to_numeric(summary_view[col], errors="coerce").round(4)
        st.subheader("Pair Summary")
        st.dataframe(summary_view, use_container_width=True, hide_index=True)

        st.subheader("Statistical Arbitrage Report")
        st.code(build_pair_report_text(pair_compare))

    st.subheader("Pair Details")
    base_symbols = [symbol for symbol in BASE_SYMBOLS if symbol in labels]
    for symbol in base_symbols:
        base = labels[symbol]
        allowed_hedges = base_hedges.get(base, [])
        if not allowed_hedges:
            title = base
        elif len(allowed_hedges) == 1:
            title = f"{base} vs {allowed_hedges[0]}"
        else:
            title = f"{base} vs {', '.join(allowed_hedges)}"
        with st.expander(title, expanded=False):
            render_tab(
                base,
                close,
                ret,
                process_var,
                obs_var,
                base_lot,
                base_side,
                interval,
                gap_hours,
                gap_action,
                use_lcm_size,
                allowed_hedges,
            )


if __name__ == "__main__":
    main()
