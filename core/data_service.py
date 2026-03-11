from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from core.data_source import LocalBondDataClient, MarketDataClient


@dataclass
class Config:
    groups: dict[str, list[dict[str, Any]]]
    settings: dict[str, Any]


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Config(groups=raw.get("groups", {}), settings=raw.get("settings", {}))


class SeasonalityDataService:
    def __init__(self, config_path: str = "config/tickers.yaml") -> None:
        self.root = Path(config_path).resolve().parent.parent
        self.config = load_config(config_path)
        self.market_data = MarketDataClient()
        self.local_bond_data = LocalBondDataClient()

    def _resolve_local_data_path(self, raw_path: str | Path) -> Path:
        path = Path(str(raw_path))
        if path.exists():
            return path

        if not path.is_absolute():
            candidate = (self.root / path).resolve()
            if candidate.exists():
                return candidate

        parts = path.parts
        marker = ("data", "investpy_bond_yields")
        for index in range(len(parts) - len(marker) + 1):
            if tuple(parts[index:index + len(marker)]) == marker:
                candidate = self.root.joinpath(*marker, *parts[index + len(marker):]).resolve()
                if candidate.exists():
                    return candidate

        return path if path.is_absolute() else (self.root / path).resolve()

    @property
    def settings(self) -> dict[str, Any]:
        return self.config.settings

    def group_names(self) -> list[str]:
        return list(self.config.groups.keys())

    def group_display_name(self, group: str) -> str:
        return str(group).replace("_", " ").title()

    def items_for_group(self, group: str) -> list[dict[str, Any]]:
        return list(self.config.groups.get(group, []))

    def all_items(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for group_items in self.config.groups.values():
            items.extend(group_items)
        return items

    def default_group(self) -> str:
        groups = self.group_names()
        if not groups:
            raise ValueError("No ticker groups found in config/tickers.yaml")
        return groups[0]

    def default_ticker(self, group: str | None = None) -> str:
        resolved_group = group or self.default_group()
        items = self.items_for_group(resolved_group)
        if not items:
            raise ValueError(f"No tickers configured for group: {resolved_group}")
        return str(items[0]["ticker"])

    def get_item(self, ticker: str) -> dict[str, Any] | None:
        for item in self.all_items():
            if item.get("ticker") == ticker:
                return item
        return None

    def display_name(self, ticker: str) -> str:
        item = self.get_item(ticker)
        if item is not None and item.get("name"):
            return str(item["name"])
        return ticker

    def default_start_date(self) -> str:
        return str(self.settings.get("default_start_date", "2000-01-01"))

    def sample_event_dates(self) -> list[str]:
        return [str(value) for value in self.settings.get("sample_event_dates", [])]

    def local_bond_data_root(self) -> Path:
        return self.root / "data" / "investpy_bond_yields"

    def local_bond_summary_path(self) -> Path:
        return self.local_bond_data_root() / "local_bond_download_summary.csv"

    def local_rates_catalog(self) -> pd.DataFrame:
        summary_path = self.local_bond_summary_path()
        if not summary_path.exists():
            return pd.DataFrame(
                columns=["country", "bond_name", "status", "history_path", "price_col", "name"]
            )

        catalog = pd.read_csv(summary_path)
        if catalog.empty:
            return catalog

        if "status" in catalog.columns:
            catalog = catalog[catalog["status"].astype(str).str.lower() == "ok"].copy()
        else:
            catalog = catalog.copy()

        if "history_path" in catalog.columns:
            catalog["history_path"] = catalog["history_path"].map(
                lambda value: str(self._resolve_local_data_path(value))
            )
            catalog = catalog[catalog["history_path"].map(lambda value: Path(str(value)).exists())].copy()

        if "request_log_path" in catalog.columns:
            catalog["request_log_path"] = catalog["request_log_path"].map(
                lambda value: str(self._resolve_local_data_path(value))
            )

        if catalog.empty:
            return catalog

        catalog["price_col"] = "Close"
        catalog["name"] = catalog["bond_name"].astype(str)

        return catalog.sort_values(["country", "bond_name"]).reset_index(drop=True)

    def local_rate_countries(self) -> list[str]:
        catalog = self.local_rates_catalog()
        if catalog.empty:
            return []
        return sorted(catalog["country"].astype(str).dropna().unique().tolist())

    def local_rates_for_country(self, country: str) -> list[dict[str, Any]]:
        catalog = self.local_rates_catalog()
        if catalog.empty:
            return []

        filtered = catalog[catalog["country"].astype(str) == str(country)].copy()
        return filtered.to_dict(orient="records")

    def get_local_rate(self, country: str, bond_name: str) -> dict[str, Any] | None:
        catalog = self.local_rates_catalog()
        if catalog.empty:
            return None

        matched = catalog[
            (catalog["country"].astype(str) == str(country))
            & (catalog["bond_name"].astype(str) == str(bond_name))
        ]
        if matched.empty:
            return None
        return matched.iloc[0].to_dict()

    def fetch_local_rate_data(
        self,
        country: str,
        bond_name: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        item = self.get_local_rate(country=country, bond_name=bond_name)
        if item is None:
            raise ValueError(f"Local rate series not found: {country} | {bond_name}")

        return self.local_bond_data.load_history(
            history_path=str(item["history_path"]),
            start=start_date,
            end=end_date,
        )

    def fetch_market_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        item = self.get_item(ticker)
        return self.market_data.download(
            ticker=ticker,
            interval=interval,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            threads=True,
        )
