from __future__ import annotations

import streamlit as st

from core.analysis import (
    build_analysis_config,
    build_calendar_year_cumulative_profile_package,
    build_cumulative_seasonality_package,
    build_custom_event_seasonality_package,
    build_daily_intra_period_profile_package,
    build_latest_snapshot_table,
    build_latest_year_monthly_comparison_package,
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
    with st.sidebar.expander("Heatmap", expanded=False):
        heatmap_demean_columns = st.checkbox(
            "Demean each month",
            value=False,
            help="Subtract each calendar month's historical average from every year so the heatmap shows deviation from the same month's norm.",
            key="heatmap_demean_columns",
        )
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
    heatmap_value_label = analysis_config["heatmap_value_label"]
    if heatmap_demean_columns:
        heatmap_value_label = f"{heatmap_value_label} vs Same-Month Avg"

    st.caption(
        f"Source: {'Investpy' if controls['category'] == 'Rates' else 'Yahoo Finance'} | "
        f"Series: {analysis_config['asset_label']}"
    )

    latest_snapshot = build_latest_snapshot_table(
        raw_data=raw_data,
        structure_label=analysis_config["asset_label"],
        price_col=analysis_config["price_col"],
        move_scale=analysis_config["value_scale"] if analysis_config["change_type"] == "difference" else None,
    )
    latest_snapshot_as_of = latest_snapshot.attrs.get("as_of_label", "N/A")
    latest_snapshot_source_label = (
        "latest available local rate observation"
        if controls["category"] == "Rates"
        else "latest available market observation"
    )
    daily_intra_period_profile = build_daily_intra_period_profile_package(
        raw_data=raw_data,
        asset_label=analysis_config["asset_label"],
        price_col=analysis_config["price_col"],
        change_type=analysis_config["change_type"],
        value_scale=analysis_config["value_scale"],
        display_unit=analysis_config["display_unit"],
        recent_year_count=3,
    )
    calendar_year_cumulative_profile = build_calendar_year_cumulative_profile_package(
        raw_data=raw_data,
        asset_label=analysis_config["asset_label"],
        price_col=analysis_config["price_col"],
        change_type=analysis_config["change_type"],
        value_scale=analysis_config["value_scale"],
        display_unit=analysis_config["display_unit"],
    )
    latest_year_monthly_comparison = build_latest_year_monthly_comparison_package(
        raw_data=raw_data,
        asset_label=analysis_config["asset_label"],
        price_col=analysis_config["price_col"],
        change_type=analysis_config["change_type"],
        value_scale=analysis_config["value_scale"],
        display_unit=analysis_config["display_unit"],
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
        value_label=heatmap_value_label,
        clip_quantile=0.95,
        minimum_clip_value=0.5,
        clip_mode="row",
        sort_descending=True,
        decimals=1,
        demean_columns=heatmap_demean_columns,
    )
    if heatmap_demean_columns:
        seasonality_heatmap["figure"].update_layout(
            title=f"{analysis_config['asset_label']} Seasonality Heatmap | Column-Demeaned"
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

    st.subheader("Latest snapshot (rolling stats)")
    st.caption(f"As of {latest_snapshot_as_of} | Based on {latest_snapshot_source_label}")
    st.dataframe(latest_snapshot.round(4), use_container_width=True)

    if analysis_config["display_unit"] == "%":
        daily_profile_header = "Daily intra-period % profile"
        cumulative_profile_header = "Cumulative % change over year"
        monthly_profile_header = "Monthly seasonality % — latest year"
    else:
        daily_profile_header = "Daily intra-period profile"
        cumulative_profile_header = "Cumulative change over year"
        monthly_profile_header = "Monthly seasonality — latest year"

    daily_profile_col, cumulative_profile_col, monthly_profile_col = st.columns(3)
    with daily_profile_col:
        st.subheader(daily_profile_header)
        st.plotly_chart(daily_intra_period_profile["figure"], use_container_width=True)
    with cumulative_profile_col:
        st.subheader(cumulative_profile_header)
        st.plotly_chart(calendar_year_cumulative_profile["figure"], use_container_width=True)
    with monthly_profile_col:
        st.subheader(monthly_profile_header)
        st.plotly_chart(latest_year_monthly_comparison["figure"], use_container_width=True)

    st.subheader("Monthly Seasonality")
    st.plotly_chart(monthly_seasonality["figure"], use_container_width=True)

    st.subheader("Cumulative Seasonality")
    st.plotly_chart(cumulative_seasonality["figure"], use_container_width=True)

    st.subheader("Seasonality Heatmap")
    if heatmap_demean_columns:
        st.caption("Column-demeaned view: each cell shows deviation from that calendar month's historical average.")
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
