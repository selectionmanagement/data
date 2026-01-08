from __future__ import annotations

import json
import random
import string
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import websocket

from symbols import short_symbol


class TradingViewError(RuntimeError):
    pass


@dataclass(frozen=True)
class ScannerSymbol:
    symbol: str
    name: str
    sector: str


_SCANNER_URL = "https://scanner.tradingview.com/thailand/scan"
_SCANNER_COLUMNS = ["name", "sector"]


def fetch_sector_catalog(*, page_size: int = 400, timeout: int = 20) -> List[ScannerSymbol]:
    page_size = max(10, int(page_size))
    offset = 0
    total_count: Optional[int] = None
    accumulated: List[ScannerSymbol] = []

    while True:
        payload = {
            "filter": [],
            "symbols": {"query": {"types": []}, "tickers": []},
            "columns": _SCANNER_COLUMNS,
            "options": {"lang": "en"},
            "sort": {"sortBy": "name", "sortOrder": "asc"},
            "range": [offset, offset + page_size - 1],
        }
        try:
            response = requests.post(_SCANNER_URL, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # noqa: BLE001
            raise TradingViewError(f"Scanner request failed: {exc}") from exc

        if not isinstance(data, dict):
            raise TradingViewError("Unexpected TradingView scanner response")

        batch = data.get("data") or []
        if total_count is None:
            total_count = int(data.get("totalCount") or 0)

        for row in batch:
            symbol = (row.get("s") or "").strip()
            values = row.get("d") or []
            name = str(values[0]) if values else short_symbol(symbol)
            sector = str(values[1]) if len(values) > 1 and values[1] is not None else ""
            accumulated.append(ScannerSymbol(symbol=symbol, name=name, sector=sector))

        offset += len(batch)
        if not batch:
            break
        if total_count and offset >= total_count:
            break
        # Protect against infinite loop if scanner stops returning data
        if len(batch) < page_size:
            break

    return accumulated


def resolution_from_label(label: str) -> str:
    label = (label or "").strip().lower()
    if label in {"weekly", "week", "w"}:
        return "W"
    if label in {"daily", "day", "d"}:
        return "D"
    if label in {"5m", "5min", "5"}:
        return "5"
    return "D"


def resolution_from_interval(interval: str) -> str:
    interval = (interval or "").strip().lower()
    mapping = {
        "1m": "1",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "1h": "60",
        "2h": "120",
        "4h": "240",
        "60m": "60",
        "1d": "D",
        "1w": "W",
        "1month": "M",
    }
    if interval in mapping:
        return mapping[interval]
    raise TradingViewError(f"Unsupported interval: {interval!r}")


def _rand_session(prefix: str) -> str:
    return prefix + "".join(random.choice(string.ascii_lowercase) for _ in range(12))


def _pack(obj: Dict) -> str:
    payload = json.dumps(obj, separators=(",", ":"))
    return f"~m~{len(payload)}~m~{payload}"


def _iter_frames(raw: str) -> List[str]:
    frames: List[str] = []
    i = 0
    while i < len(raw):
        if raw.startswith("~h~", i):
            j = raw.find("~m~", i)
            if j == -1:
                frames.append(raw[i:])
                break
            frames.append(raw[i:j])
            i = j
            continue

        if not raw.startswith("~m~", i):
            j = raw.find("~m~", i)
            if j == -1:
                break
            i = j
            continue

        i += 3
        j = raw.find("~m~", i)
        if j == -1:
            break
        length_str = raw[i:j]
        try:
            length = int(length_str)
        except ValueError:
            break
        i = j + 3
        payload = raw[i : i + length]
        i += length
        frames.append(payload)
    return frames


@dataclass
class TradingViewClient:
    url: str = "wss://data.tradingview.com/socket.io/websocket"
    timeout: int = 20

    def _connect(self) -> websocket.WebSocket:
        headers = [
            "Origin: https://www.tradingview.com",
            "User-Agent: Mozilla/5.0",
        ]
        return websocket.create_connection(self.url, header=headers, timeout=self.timeout, enable_multithread=True)

    def _send(self, ws: websocket.WebSocket, obj: Dict) -> None:
        ws.send(_pack(obj))

    def get_ohlcv(
        self,
        *,
        symbol: str,
        resolution: str,
        bars: int,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        max_bars_per_request: int = 5000,
        max_batches: int = 200,
        sleep_seconds: float = 0.2,
        progress: bool = False,
        progress_label: Optional[str] = None,
    ) -> pd.DataFrame:
        symbol = (symbol or "").strip().upper()
        if not symbol or ":" not in symbol:
            raise TradingViewError(f"Invalid symbol: {symbol!r} (expected like 'SET:ADVANC')")

        resolution = (resolution or "").strip().upper()
        allowed = {"D", "W", "M", "1", "5", "15", "30", "60", "120", "240"}
        if resolution not in allowed:
            raise TradingViewError(f"Unsupported resolution: {resolution!r}")

        bars = int(bars)
        if bars <= 0:
            raise TradingViewError("bars must be > 0")

        if start is not None:
            start = pd.Timestamp(start)
            if start.tzinfo is None:
                start = start.tz_localize("UTC")
            else:
                start = start.tz_convert("UTC")
        if end is not None:
            end = pd.Timestamp(end)
            if end.tzinfo is None:
                end = end.tz_localize("UTC")
            else:
                end = end.tz_convert("UTC")

        bars_per_request = min(max_bars_per_request, max(1, bars))
        max_batches = max(1, int(max_batches))

        def _log(message: str) -> None:
            if progress:
                print(message, flush=True)

        def _fmt_epoch(epoch: Optional[int]) -> str:
            if epoch is None:
                return "n/a"
            ts = pd.to_datetime(epoch, unit="s", utc=True)
            return ts.strftime("%Y-%m-%d %H:%M UTC")

        def _store_row(store: Dict[int, Tuple[float, float, float, float, float]], t: int, vals: List) -> None:
            if not vals or len(vals) < 5:
                return
            store[int(t)] = (
                float(vals[1]) if vals[1] is not None else float("nan"),
                float(vals[2]) if vals[2] is not None else float("nan"),
                float(vals[3]) if vals[3] is not None else float("nan"),
                float(vals[4]) if vals[4] is not None else float("nan"),
                float(vals[5]) if len(vals) > 5 and vals[5] is not None else float("nan"),
            )

        def _parse_timescale(store: Dict[int, Tuple[float, float, float, float, float]], payload: Dict) -> None:
            s1 = payload.get("s1")
            if not isinstance(s1, dict):
                return

            rows = s1.get("s")
            if isinstance(rows, list) and rows:
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    vals = row.get("v")
                    if not isinstance(vals, list) or len(vals) < 5:
                        continue
                    try:
                        t = int(float(vals[0]))
                    except Exception:
                        continue
                    _store_row(store, t, vals)
                return

            t_arr = s1.get("t")
            if not isinstance(t_arr, list) or not t_arr:
                return
            n = len(t_arr)

            def _norm(arr: Optional[List[float]]) -> List[float]:
                if not isinstance(arr, list) or not arr:
                    return [float("nan")] * n
                if len(arr) < n:
                    return list(arr) + [float("nan")] * (n - len(arr))
                if len(arr) > n:
                    return list(arr[:n])
                return list(arr)

            o = _norm(s1.get("o"))
            h = _norm(s1.get("h"))
            l = _norm(s1.get("l"))
            c = _norm(s1.get("c"))
            v = _norm(s1.get("v"))
            for i, t in enumerate(t_arr):
                try:
                    ts_val = int(float(t))
                except Exception:
                    continue
                _store_row(store, ts_val, [ts_val, o[i], h[i], l[i], c[i], v[i]])

        def _recv_until_complete(ws: websocket.WebSocket, store: Dict[int, Tuple[float, float, float, float, float]]) -> None:
            start_wait = time.time()
            while True:
                if time.time() - start_wait > self.timeout:
                    raise TradingViewError(f"Timeout while fetching {symbol} ({resolution})")

                raw = ws.recv()
                for frame in _iter_frames(raw):
                    if frame.startswith("~h~"):
                        ws.send(f"~m~{len(frame)}~m~{frame}")
                        continue

                    try:
                        msg = json.loads(frame)
                    except json.JSONDecodeError:
                        continue

                    m = msg.get("m")
                    if m == "timescale_update":
                        payload = msg.get("p", [None, None])[1] or {}
                        if isinstance(payload, dict):
                            _parse_timescale(store, payload)
                    if m == "series_completed":
                        return

        ws = self._connect()
        try:
            try:
                ws.recv()
            except Exception:
                pass

            chart_session = _rand_session("cs_")
            quote_session = _rand_session("qs_")

            self._send(ws, {"m": "set_auth_token", "p": ["unauthorized_user_token"]})
            self._send(ws, {"m": "chart_create_session", "p": [chart_session, ""]})
            self._send(ws, {"m": "quote_create_session", "p": [quote_session]})
            self._send(
                ws,
                {
                    "m": "quote_set_fields",
                    "p": [
                        quote_session,
                        "ch",
                        "chp",
                        "lp",
                        "volume",
                        "short_name",
                        "exchange",
                        "description",
                        "type",
                    ],
                },
            )
            self._send(ws, {"m": "quote_add_symbols", "p": [quote_session, symbol]})
            self._send(
                ws,
                {
                    "m": "resolve_symbol",
                    "p": [
                        chart_session,
                        "symbol_1",
                        f'={{"symbol":"{symbol}","adjustment":"splits","session":"regular"}}',
                    ],
                },
            )
            self._send(
                ws,
                {"m": "create_series", "p": [chart_session, "s1", "s1", "symbol_1", resolution, bars_per_request]},
            )
            self._send(ws, {"m": "switch_timezone", "p": [chart_session, "Etc/UTC"]})

            data: Dict[int, Tuple[float, float, float, float, float]] = {}
            _recv_until_complete(ws, data)

            min_ts = min(data.keys()) if data else None
            max_ts = max(data.keys()) if data else None
            start_epoch = int(start.timestamp()) if start is not None else None
            target_bars = bars if start is None and end is None and bars > bars_per_request else None
            batches = 1
            label = progress_label or symbol
            _log(
                f"[TV] {label} batch {batches} bars={len(data)} "
                f"range={_fmt_epoch(min_ts)}..{_fmt_epoch(max_ts)}"
            )

            while batches < max_batches:
                need_more = False
                if start_epoch is not None and min_ts is not None and min_ts > start_epoch:
                    need_more = True
                if target_bars is not None and len(data) < target_bars:
                    need_more = True
                if not need_more:
                    break

                self._send(ws, {"m": "request_more_data", "p": [chart_session, "s1", bars_per_request]})
                _recv_until_complete(ws, data)
                new_min = min(data.keys()) if data else None
                if new_min is None or (min_ts is not None and new_min >= min_ts):
                    break
                min_ts = new_min
                batches += 1
                _log(
                    f"[TV] {label} batch {batches} bars={len(data)} "
                    f"range={_fmt_epoch(min_ts)}..{_fmt_epoch(max_ts)}"
                )
                if sleep_seconds > 0:
                    time.sleep(float(sleep_seconds))

            if start_epoch is not None and min_ts is not None and min_ts > start_epoch:
                _log(f"[TV] {label} stopped at batch {batches} (start not reached)")

            if not data:
                return pd.DataFrame()

            df = pd.DataFrame.from_dict(
                data,
                orient="index",
                columns=["open", "high", "low", "close", "volume"],
            )
            df.index = pd.to_datetime(df.index.astype("int64"), unit="s", utc=True)
            df = df.sort_index()
            if start is not None:
                df = df[df.index >= start]
            if end is not None:
                df = df[df.index <= end]
            return df
        except TradingViewError:
            raise
        except Exception as e:  # noqa: BLE001
            raise TradingViewError(str(e)) from e
        finally:
            try:
                ws.close()
            except Exception:
                pass
