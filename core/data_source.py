from __future__ import annotations

from pathlib import Path
import random
import time
from typing import Iterable

import pandas as pd
import yfinance as yf


class YFClient:
    def __init__(self, max_retries: int = 4, base_sleep: float = 0.8) -> None:
        self.max_retries = max_retries
        self.base_sleep = base_sleep

    def download(
        self,
        tickers: Iterable[str],
        interval: str = "1d",
        period: str | None = None,
        start: str | None = None,
        end: str | None = None,
        auto_adjust: bool = False,
        threads: bool = True,
    ) -> pd.DataFrame:
        ticker_list = list(tickers)
        if not ticker_list:
            raise ValueError("tickers list is empty")

        last_err: Exception | None = None
        for k in range(self.max_retries):
            try:
                df = yf.download(
                    tickers=" ".join(ticker_list),
                    interval=interval,
                    period=period,
                    start=start,
                    end=end,
                    auto_adjust=auto_adjust,
                    threads=threads,
                    progress=False,
                    group_by="column",
                )
                if df is None or df.empty:
                    raise ValueError("yfinance returned empty dataframe")
                return df
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                sleep = self.base_sleep * (2**k) + random.random() * 0.2
                time.sleep(sleep)

        raise RuntimeError(f"yfinance download failed after retries: {last_err}")


class LocalBondDataClient:
    def load_history(
        self,
        history_path: str | Path,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        path = Path(history_path)
        if not path.exists():
            raise FileNotFoundError(f"Local bond history file not found: {path}")

        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if df is None or df.empty:
            raise ValueError(f"Local bond history is empty: {path}")

        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()].sort_index()

        if start is not None:
            df = df.loc[df.index >= pd.Timestamp(start)]
        if end is not None:
            df = df.loc[df.index <= pd.Timestamp(end)]

        if df.empty:
            raise ValueError("No local rate observations are available for the selected date range.")

        return df


class MarketDataClient:
    def __init__(self) -> None:
        self.yf = YFClient()

    def download(
        self,
        ticker: str,
        interval: str = "1d",
        period: str | None = None,
        start: str | None = None,
        end: str | None = None,
        auto_adjust: bool = False,
        threads: bool = True,
    ) -> pd.DataFrame:
        return self.yf.download(
            tickers=[ticker],
            interval=interval,
            period=period,
            start=start,
            end=end,
            auto_adjust=auto_adjust,
            threads=threads,
        )
