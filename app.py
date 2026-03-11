from __future__ import annotations

import streamlit as st

from core.analysis import (
    build_analysis_config,
    build_cumulative_seasonality_package,
    build_custom_event_seasonality_package,
    build_monthly_seasonality_package,
    build_seasonality_heatmap_package,
)
from core.ui import (
    get_service,
    load_market_data,
    render_common_sidebar,
    render_event_controls,
    show_market_data_preview,
)


st.set_page_config(page_title="Seasonality Dashboard", layout="wide")


def main() -> None:
    st.title("Seasonality Dashboard")
    service = get_service()

    with st.sidebar:
        if st.button("Refresh cached data", type="primary"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.toast("Cached downloads cleared.")

    controls = render_common_sidebar(service)
    event_controls = render_event_controls(service.sample_event_dates())
    raw_data = load_market_data(
        category=controls["category"],
        ticker=controls["ticker"],
        start_date=controls["start_date"],
        end_date=controls["end_date"],
        rate_country=controls["rate_country"],
        rate_bond_name=controls["rate_bond_name"],
    )

    analysis_ticker = controls["ticker"] or controls["rate_bond_name"] or "Asset"
    resolved_asset_class_override = controls["asset_class_override"]
    if controls["category"] == "Rates" and resolved_asset_class_override is None:
        resolved_asset_class_override = "bond"

    resolved_asset_label_override = controls["asset_label_override"]
    if controls["category"] == "Rates" and resolved_asset_label_override is None:
        resolved_asset_label_override = controls["selected_display_name"]

    analysis_config = build_analysis_config(
        ticker=analysis_ticker,
        asset_class_override=resolved_asset_class_override,
        asset_label_override=resolved_asset_label_override,
        price_col=controls["price_col_override"],
        change_type_override=controls["change_type_override"],
        display_unit_override=controls["display_unit_override"],
        value_scale_override=controls["value_scale_override"],
    )

    st.caption(
        f"Source: {'Local rates data' if controls['category'] == 'Rates' else 'Yahoo Finance'} | "
        f"Series: {analysis_config['asset_label']}"
    )

    monthly_seasonality = build_monthly_seasonality_package(
        raw_data=raw_data,
        asset_label=analysis_config["asset_label"],
        price_col=analysis_config["price_col"],
        change_type=analysis_config["change_type"],
        drop_incomplete_last_month=True,
        highlight_year=None,
        value_scale=analysis_config["value_scale"],
        y_axis_title=analysis_config["monthly_y_axis_title"],
    )
    cumulative_seasonality = build_cumulative_seasonality_package(
        raw_data=raw_data,
        asset_label=analysis_config["asset_label"],
        price_col=analysis_config["price_col"],
        change_type=analysis_config["change_type"],
        drop_incomplete_last_month=True,
        highlight_year=None,
        value_scale=analysis_config["value_scale"],
        y_axis_title=analysis_config["cumulative_y_axis_title"],
    )
    seasonality_heatmap = build_seasonality_heatmap_package(
        raw_data=raw_data,
        asset_label=analysis_config["asset_label"],
        price_col=analysis_config["price_col"],
        change_type=analysis_config["change_type"],
        drop_incomplete_last_month=True,
        value_scale=analysis_config["value_scale"],
        display_unit=analysis_config["display_unit"],
        value_label=analysis_config["heatmap_value_label"],
        clip_quantile=0.95,
        minimum_clip_value=0.5,
        clip_mode="row",
        sort_descending=True,
        decimals=1,
    )
    custom_event_title = f"{analysis_config['asset_label']} Custom Event Seasonality | User Event Dates"
    custom_event_seasonality = build_custom_event_seasonality_package(
        raw_data=raw_data,
        event_dates=event_controls["event_dates"],
        asset_label=analysis_config["asset_label"],
        price_col=analysis_config["price_col"],
        change_type=analysis_config["change_type"],
        window_size=event_controls["window_size"],
        alignment=event_controls["alignment"],
        require_complete_window=event_controls["require_complete_window"],
        value_scale=analysis_config["value_scale"],
        display_unit=analysis_config["display_unit"],
        title_label=custom_event_title,
        current_label="Most Recent Event",
        table_decimals=1,
        table_clip_quantile=0.95,
        table_minimum_clip_value=0.5,
        table_clip_mode="row",
    )

    st.subheader("Monthly Seasonality")
    st.plotly_chart(monthly_seasonality["figure"], use_container_width=True)

    st.subheader("Cumulative Seasonality")
    st.plotly_chart(cumulative_seasonality["figure"], use_container_width=True)

    st.subheader("Seasonality Heatmap")
    st.plotly_chart(seasonality_heatmap["figure"], use_container_width=True)

    st.subheader("Custom Event Seasonality")
    st.plotly_chart(custom_event_seasonality["event_heatmap_figure"], use_container_width=True)
    st.plotly_chart(custom_event_seasonality["figure"], use_container_width=True)

    with st.expander("Custom event diagnostics", expanded=False):
        st.dataframe(custom_event_seasonality["event_metadata"], use_container_width=True)
        st.dataframe(custom_event_seasonality["summary_stats"].round(2), use_container_width=True)

    show_market_data_preview(raw_data)


if __name__ == "__main__":
    main()
