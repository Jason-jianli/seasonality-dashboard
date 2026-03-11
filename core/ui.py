from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from core.data_service import SeasonalityDataService


@st.cache_resource
def get_service() -> SeasonalityDataService:
    service = SeasonalityDataService(config_path="config/tickers.yaml")
    required_methods = (
        "local_rate_countries",
        "local_rates_for_country",
        "get_local_rate",
        "fetch_local_rate_data",
    )
    missing_methods = [method_name for method_name in required_methods if not hasattr(service, method_name)]
    if missing_methods:
        raise AttributeError(
            "SeasonalityDataService is missing required methods: "
            + ", ".join(missing_methods)
        )
    return service


@st.cache_data(show_spinner=False, ttl=300)
def load_market_data(
    category: str,
    ticker: str | None,
    start_date: str,
    end_date: str,
    rate_country: str | None = None,
    rate_bond_name: str | None = None,
) -> pd.DataFrame:
    svc = get_service()
    if category == "Rates":
        if not rate_country or not rate_bond_name:
            raise ValueError("Please select a local rate country and bond series.")
        return svc.fetch_local_rate_data(
            country=rate_country,
            bond_name=rate_bond_name,
            start_date=start_date,
            end_date=end_date,
        )

    if not ticker:
        raise ValueError("Please enter a Yahoo Finance ticker.")

    return svc.fetch_market_data(ticker=ticker, start_date=start_date, end_date=end_date)


def _blank_to_none(value: str) -> str | None:
    stripped = str(value).strip()
    return stripped or None


def render_common_sidebar(service: SeasonalityDataService) -> dict[str, Any]:
    st.sidebar.header("Controls")

    if not hasattr(service, "local_rate_countries"):
        st.cache_resource.clear()
        service = SeasonalityDataService(config_path="config/tickers.yaml")

    category_options = ["Equities & FX & Others", "Rates"]
    selected_category = st.sidebar.selectbox(
        "Category",
        options=category_options,
        index=category_options.index(st.session_state.get("selected_category", category_options[0]))
        if st.session_state.get("selected_category", category_options[0]) in category_options
        else 0,
        key="selected_category",
    )

    selected_ticker: str | None = None
    selected_rate_country: str | None = None
    selected_rate_bond_name: str | None = None
    selected_item: dict[str, Any] = {}

    if selected_category == "Rates":
        available_countries = service.local_rate_countries()
        if not available_countries:
            raise ValueError("No local rate files were found in data/investpy_bond_yields.")

        default_country = st.session_state.get("selected_rate_country", available_countries[0])
        if default_country not in available_countries:
            default_country = available_countries[0]

        selected_rate_country = st.sidebar.selectbox(
            "Country",
            options=available_countries,
            index=available_countries.index(default_country),
            key="selected_rate_country",
        )

        rate_items = service.local_rates_for_country(selected_rate_country)
        rate_bond_options = [str(item["bond_name"]) for item in rate_items]
        if not rate_bond_options:
            raise ValueError(f"No local rate files were found for {selected_rate_country}.")

        default_bond = st.session_state.get("selected_rate_bond_name", rate_bond_options[0])
        if default_bond not in rate_bond_options:
            default_bond = rate_bond_options[0]

        selected_rate_bond_name = st.sidebar.selectbox(
            "Local rate series",
            options=rate_bond_options,
            index=rate_bond_options.index(default_bond),
            key="selected_rate_bond_name",
        )
        selected_item = service.get_local_rate(selected_rate_country, selected_rate_bond_name) or {}
    else:
        selected_ticker = st.sidebar.text_input(
            "Ticker",
            value=st.session_state.get("selected_ticker", "USDJPY=X"),
            key="selected_ticker",
            help="Enter any Yahoo Finance ticker. Example: USDJPY=X, AAPL, GC=F.",
        ).strip()
        selected_item = service.get_item(selected_ticker) or {}

    default_start = pd.Timestamp(service.default_start_date()).date()
    default_end = pd.Timestamp(datetime.now().date())

    start_date_value = st.sidebar.date_input(
        "Start date",
        value=st.session_state.get("start_date", default_start),
        key="start_date",
    )
    end_date_value = st.sidebar.date_input(
        "End date",
        value=st.session_state.get("end_date", default_end.date()),
        key="end_date",
    )

    with st.sidebar.expander("Advanced overrides", expanded=False):
        asset_class_override = st.selectbox(
            "Asset class override",
            options=[None, "bond", "other"],
            format_func=lambda value: "Auto" if value is None else value,
            key="asset_class_override",
        )
        asset_label_override_raw = st.text_input(
            "Asset label override",
            value=st.session_state.get("asset_label_override", ""),
            key="asset_label_override",
        )
        price_col_override = st.text_input(
            "Price column",
            value=str(selected_item.get("price_col", "Close")),
            key="price_col_override",
        )
        change_type_override = st.selectbox(
            "Change type override",
            options=[None, "pct_change", "difference"],
            format_func=lambda value: "Auto" if value is None else value,
            key="change_type_override",
        )
        display_unit_override_raw = st.text_input(
            "Display unit override",
            value=st.session_state.get("display_unit_override", ""),
            key="display_unit_override",
        )
        value_scale_override_raw = st.text_input(
            "Value scale override",
            value=st.session_state.get("value_scale_override", ""),
            key="value_scale_override",
            help="Leave blank to keep the notebook default logic.",
        )

    return {
        "category": selected_category,
        "ticker": selected_ticker,
        "rate_country": selected_rate_country,
        "rate_bond_name": selected_rate_bond_name,
        "selected_display_name": str(selected_item.get("name") or selected_rate_bond_name or selected_ticker or "Asset"),
        "start_date": pd.Timestamp(start_date_value).strftime("%Y-%m-%d"),
        "end_date": pd.Timestamp(end_date_value).strftime("%Y-%m-%d"),
        "asset_class_override": asset_class_override,
        "asset_label_override": _blank_to_none(asset_label_override_raw),
        "price_col_override": price_col_override.strip() or "Close",
        "change_type_override": change_type_override,
        "display_unit_override": _blank_to_none(display_unit_override_raw),
        "value_scale_override": float(value_scale_override_raw) if value_scale_override_raw.strip() else None,
    }


def render_event_controls(default_event_dates: list[str]) -> dict[str, Any]:
    st.sidebar.header("Event Study")
    event_text_default = "\n".join(default_event_dates)
    event_dates_text = st.sidebar.text_area(
        "Event dates",
        value=st.session_state.get("event_dates_text", event_text_default),
        height=220,
        key="event_dates_text",
        help="One date per line, or separate dates with commas / semicolons.",
    )
    window_size = st.sidebar.slider("Window size", min_value=3, max_value=30, value=10, step=1)
    alignment = st.sidebar.selectbox("Alignment", options=["next", "previous", "nearest"], index=0)
    require_complete_window = st.sidebar.checkbox("Require complete window", value=True)

    return {
        "event_dates": parse_event_dates_text(event_dates_text),
        "window_size": int(window_size),
        "alignment": alignment,
        "require_complete_window": require_complete_window,
    }


def parse_event_dates_text(raw_text: str) -> list[str]:
    tokens = re.split(r"[\n,;]+", str(raw_text))
    values = [token.strip() for token in tokens if token.strip()]
    if not values:
        raise ValueError("Please enter at least one valid event date.")
    return values


def show_analysis_config(analysis_config: dict[str, Any]) -> None:
    config_df = pd.Series(analysis_config, name="value").to_frame()
    st.dataframe(config_df, use_container_width=True)


def show_market_data_preview(raw_data: pd.DataFrame) -> None:
    with st.expander("Raw market data preview", expanded=False):
        st.dataframe(raw_data.tail(20), use_container_width=True)
