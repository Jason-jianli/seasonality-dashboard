from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
	"Config",
	"SeasonalityDataService",
	"load_config",
	"LocalBondDataClient",
	"MarketDataClient",
	"YFClient",
]


def __getattr__(name: str) -> Any:
	if name in {"Config", "SeasonalityDataService", "load_config"}:
		module = import_module("core.data_service")
		return getattr(module, name)

	if name in {"LocalBondDataClient", "MarketDataClient", "YFClient"}:
		module = import_module("core.data_source")
		return getattr(module, name)

	raise AttributeError(f"module 'core' has no attribute {name!r}")
