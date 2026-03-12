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
        self._local_rates_catalog_cache: pd.DataFrame | None = None

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

    def _resolve_optional_local_data_path(self, raw_path: Any) -> str | None:
        if raw_path is None or pd.isna(raw_path):
            return None

        text = str(raw_path).strip()
        if not text:
            return None

        return str(self._resolve_local_data_path(text))

    def _finalize_local_rates_catalog(self, catalog: pd.DataFrame) -> pd.DataFrame:
        columns = ["country", "bond_name", "status", "history_path", "request_log_path", "price_col", "name"]
        if catalog is None or catalog.empty:
            return pd.DataFrame(columns=columns)

        catalog = catalog.copy()
        required_columns = ["country", "bond_name", "history_path"]
        for column in required_columns:
            if column not in catalog.columns:
                return pd.DataFrame(columns=columns)

        if "status" not in catalog.columns:
            catalog["status"] = "ok"
        catalog = catalog[catalog["status"].astype(str).str.lower() == "ok"].copy()
        if catalog.empty:
            return pd.DataFrame(columns=columns)

        catalog["country"] = catalog["country"].astype(str).str.strip()
        catalog["bond_name"] = catalog["bond_name"].astype(str).str.strip()
        catalog["history_path"] = catalog["history_path"].map(self._resolve_optional_local_data_path)
        catalog = catalog.dropna(subset=["country", "bond_name", "history_path"]).copy()
        catalog = catalog[catalog["history_path"].map(lambda value: Path(str(value)).exists())].copy()
        if catalog.empty:
            return pd.DataFrame(columns=columns)

        if "request_log_path" not in catalog.columns:
            catalog["request_log_path"] = None
        else:
            catalog["request_log_path"] = catalog["request_log_path"].map(self._resolve_optional_local_data_path)

        if "price_col" not in catalog.columns:
            catalog["price_col"] = "Close"
        else:
            catalog["price_col"] = catalog["price_col"].fillna("Close").astype(str)

        if "name" not in catalog.columns:
            catalog["name"] = catalog["bond_name"]
        else:
            catalog["name"] = catalog["name"].fillna(catalog["bond_name"]).astype(str)

        return catalog.loc[:, columns]

    def _fallback_bond_name(self, country: str, history_path: Path) -> str:
        country_label = str(country).replace("_", " ").title()
        stem = history_path.stem
        country_slug = str(country).replace(" ", "_")
        suffix = stem[len(country_slug) + 1:] if stem.startswith(f"{country_slug}_") else stem

        display_parts: list[str] = []
        for part in suffix.split("_"):
            if not part:
                continue
            if any(char.isdigit() for char in part) or len(part) <= 2:
                display_parts.append(part.upper())
            else:
                display_parts.append(part.title())

        return f"{country_label} {' '.join(display_parts)}".strip()

    def _discover_local_rates_catalog(self) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        data_root = self.local_bond_data_root()
        if not data_root.exists():
            return pd.DataFrame()

        for country_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
            for history_path in sorted(country_dir.glob("*.csv")):
                if history_path.name.endswith("_request_log.csv"):
                    continue

                request_log_path = history_path.with_name(f"{history_path.stem}_request_log.csv")
                country = country_dir.name.replace("_", " ")
                bond_name = self._fallback_bond_name(country, history_path)

                if request_log_path.exists():
                    try:
                        request_log = pd.read_csv(request_log_path)
                    except Exception:  # noqa: BLE001
                        request_log = pd.DataFrame()

                    if not request_log.empty:
                        if "country" in request_log.columns:
                            country_values = request_log["country"].dropna().astype(str).str.strip()
                            if not country_values.empty:
                                country = country_values.iloc[0]
                        if "bond_name" in request_log.columns:
                            bond_values = request_log["bond_name"].dropna().astype(str).str.strip()
                            if not bond_values.empty:
                                bond_name = bond_values.iloc[0]

                records.append(
                    {
                        "country": country,
                        "bond_name": bond_name,
                        "status": "ok",
                        "history_path": str(history_path),
                        "request_log_path": str(request_log_path) if request_log_path.exists() else None,
                    }
                )

        return pd.DataFrame.from_records(records)

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

    def local_rates_catalog(self) -> pd.DataFrame:
        if self._local_rates_catalog_cache is not None:
            return self._local_rates_catalog_cache.copy()

        discovered_catalog = self._finalize_local_rates_catalog(self._discover_local_rates_catalog())

        catalog = discovered_catalog.sort_values(["country", "bond_name"]).reset_index(drop=True)
        self._local_rates_catalog_cache = catalog.copy()
        return catalog

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
