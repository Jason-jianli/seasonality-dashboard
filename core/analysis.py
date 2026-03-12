from __future__ import annotations

import calendar
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

MONTH_NUMBERS = list(range(1, 13))
MONTH_LABELS = [calendar.month_abbr[month] for month in MONTH_NUMBERS]

DEFAULT_BOND_KEYWORDS = (
    "BOND",
    "YIELD",
    "RATE",
    "TREASURY",
    "NOTE",
    "BUND",
    "GILT",
    "JGB",
    "SWAP",
    "SOFR",
    "ESTR",
    "SONIA",
    "EURIBOR",
)
DEFAULT_BOND_TICKER_HINTS = (
    "^TNX",
    "^IRX",
    "^FVX",
    "^TYX",
    "US02Y",
    "US05Y",
    "US10Y",
    "US30Y",
)


def infer_asset_class(ticker: str, asset_class_override: Optional[str] = None) -> str:
    """Infer whether the selected asset should be treated as a bond/rate or as a percent-return asset."""
    if asset_class_override is not None:
        resolved_override = str(asset_class_override).strip().lower()
        if resolved_override not in {"bond", "other"}:
            raise ValueError("asset_class_override must be either 'bond', 'other', or None.")
        return resolved_override

    normalized_ticker = str(ticker).upper().strip()
    if normalized_ticker in DEFAULT_BOND_TICKER_HINTS:
        return "bond"

    if any(keyword in normalized_ticker for keyword in DEFAULT_BOND_KEYWORDS):
        return "bond"

    return "other"


def normalize_asset_label(ticker: str, asset_label_override: Optional[str] = None) -> str:
    """Create a clean display label from a Yahoo Finance ticker."""
    if asset_label_override is not None and str(asset_label_override).strip():
        return str(asset_label_override).strip()

    label = str(ticker).strip()
    for suffix in ("=X", "=F"):
        if label.endswith(suffix):
            label = label[: -len(suffix)]
    label = label.replace("^", "").replace("_", " ").strip()

    return label or "Asset"


def build_analysis_config(
    ticker: str,
    asset_class_override: Optional[str] = None,
    asset_label_override: Optional[str] = None,
    price_col: str = "Close",
    change_type_override: Optional[str] = None,
    display_unit_override: Optional[str] = None,
    value_scale_override: Optional[float] = None,
) -> Dict[str, Any]:
    """Build a single source of truth so every downstream module follows the same asset logic."""
    asset_class = infer_asset_class(ticker=ticker, asset_class_override=asset_class_override)
    asset_label = normalize_asset_label(ticker=ticker, asset_label_override=asset_label_override)

    if change_type_override is not None:
        change_type = str(change_type_override).strip()
    else:
        change_type = "difference" if asset_class == "bond" else "pct_change"

    if change_type not in {"pct_change", "difference"}:
        raise ValueError("change_type_override must be either 'pct_change', 'difference', or None.")

    if display_unit_override is not None:
        display_unit = str(display_unit_override).strip()
    else:
        display_unit = "bp" if asset_class == "bond" else "%"

    if value_scale_override is not None:
        value_scale = float(value_scale_override)
    elif change_type == "difference" and display_unit.lower() == "bp":
        value_scale = 100.0
    else:
        value_scale = None

    if display_unit == "%":
        monthly_y_axis_title = "Monthly Return (%)" if change_type == "pct_change" else "Monthly Change (%)"
        cumulative_y_axis_title = "Cumulative Return (%)" if change_type == "pct_change" else "Cumulative Change (%)"
        heatmap_value_label = "Monthly Return (%)" if change_type == "pct_change" else "Monthly Change (%)"
        event_y_axis_title = "Relative Move (%)" if change_type == "pct_change" else "Relative Change (%)"
    elif display_unit.lower() == "bp":
        monthly_y_axis_title = "Monthly Change (bp)"
        cumulative_y_axis_title = "Cumulative Change (bp)"
        heatmap_value_label = "Monthly Change (bp)"
        event_y_axis_title = "Relative Change (bp)"
    else:
        monthly_y_axis_title = "Monthly Change"
        cumulative_y_axis_title = "Cumulative Change"
        heatmap_value_label = "Monthly Change"
        event_y_axis_title = "Relative Change"

    return {
        "ticker": ticker,
        "asset_class": asset_class,
        "asset_label": asset_label,
        "price_col": price_col,
        "change_type": change_type,
        "display_unit": display_unit,
        "value_scale": value_scale,
        "monthly_y_axis_title": monthly_y_axis_title,
        "cumulative_y_axis_title": cumulative_y_axis_title,
        "heatmap_value_label": heatmap_value_label,
        "event_y_axis_title": event_y_axis_title,
    }


def flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance columns so downstream logic works for both flat and MultiIndex outputs."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")

    output = df.copy()

    if isinstance(output.columns, pd.MultiIndex):
        flattened_columns = []
        for column_tuple in output.columns.to_flat_index():
            parts = [
                str(part)
                for part in column_tuple
                if str(part).strip() and str(part).lower() != "nan"
            ]
            flattened_columns.append("_".join(parts))
        output.columns = flattened_columns
    else:
        output.columns = [str(column) for column in output.columns]

    return output


def resolve_price_column(columns: pd.Index, preferred: str = "Close") -> str:
    """Resolve the most appropriate price column from a yfinance-style dataset."""
    normalized_map = {
        str(column).lower().replace(" ", "").replace("_", ""): str(column)
        for column in columns
    }

    candidate_keys = [
        preferred,
        "Close",
        "Adj Close",
        "Adj_Close",
        "close",
        "adjclose",
    ]

    for candidate in candidate_keys:
        normalized_candidate = candidate.lower().replace(" ", "").replace("_", "")
        if normalized_candidate in normalized_map:
            return normalized_map[normalized_candidate]

    for prefix in ("close", "adjclose"):
        matching_columns = [
            str(column)
            for column in columns
            if str(column).lower().replace(" ", "").startswith(prefix)
        ]
        if matching_columns:
            return matching_columns[0]

    raise KeyError(
        "Could not find a usable price column. Available columns: "
        f"{list(columns)}"
    )


def prepare_price_series(raw_data: pd.DataFrame, price_col: str = "Close") -> pd.Series:
    """Validate raw market data and return a clean price series."""
    if raw_data is None or raw_data.empty:
        raise ValueError("Input market data is empty. Pull data before running seasonality analysis.")

    data = flatten_yfinance_columns(raw_data)

    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index, errors="coerce")

    data = data[~data.index.isna()].sort_index()

    if getattr(data.index, "tz", None) is not None:
        data.index = data.index.tz_localize(None)

    data = data[~data.index.duplicated(keep="last")]

    resolved_price_col = resolve_price_column(data.columns, preferred=price_col)
    price_series = pd.to_numeric(data[resolved_price_col], errors="coerce").dropna()

    if price_series.empty:
        raise ValueError("The resolved price series is empty after cleaning.")

    if price_series.shape[0] < 60:
        raise ValueError("At least 60 daily observations are recommended for monthly seasonality.")

    price_series.name = "price"
    return price_series


def build_latest_snapshot_table(
    raw_data: pd.DataFrame,
    structure_label: str = "Asset",
    price_col: str = "Close",
    move_scale: Optional[float] = None,
    one_month_window: int = 21,
    three_month_window: int = 63,
    move_window: int = 62,
    history_window: int = 252,
) -> pd.DataFrame:
    """Build a one-row rolling statistics snapshot table for the latest observation."""
    price_series = prepare_price_series(raw_data=raw_data, price_col=price_col)
    latest_timestamp = pd.Timestamp(price_series.index[-1])

    latest_level = float(price_series.iloc[-1])

    one_month_slice = price_series.tail(min(one_month_window, len(price_series)))
    three_month_slice = price_series.tail(min(three_month_window, len(price_series)))

    one_month_avg = float(one_month_slice.mean()) if not one_month_slice.empty else np.nan
    three_month_avg = float(three_month_slice.mean()) if not three_month_slice.empty else np.nan
    three_month_high = float(three_month_slice.max()) if not three_month_slice.empty else np.nan
    three_month_low = float(three_month_slice.min()) if not three_month_slice.empty else np.nan

    if pd.notna(three_month_high) and pd.notna(three_month_low) and three_month_high != three_month_low:
        three_month_pct = ((latest_level - three_month_low) / (three_month_high - three_month_low)) * 100.0
    else:
        three_month_pct = np.nan

    difference_scale = 1.0 if move_scale is None else float(move_scale)
    change_vs_one_month_avg = (latest_level - one_month_avg) * difference_scale if pd.notna(one_month_avg) else np.nan

    daily_changes = price_series.diff().dropna()
    three_month_daily_changes = daily_changes.tail(min(three_month_window, len(daily_changes)))
    avg_three_month_daily_change = (
        float(three_month_daily_changes.mean()) * difference_scale
        if not three_month_daily_changes.empty
        else np.nan
    )
    three_month_sigma = (
        float(three_month_daily_changes.std()) * difference_scale
        if not three_month_daily_changes.empty
        else np.nan
    )

    latest_move = (
        float(price_series.iloc[-1] - price_series.iloc[-(move_window + 1)]) * difference_scale
        if len(price_series) > move_window
        else np.nan
    )

    rolling_moves = price_series.diff(periods=move_window).dropna() * difference_scale
    rolling_moves_window = rolling_moves.tail(min(history_window, len(rolling_moves)))

    avg_move = float(rolling_moves_window.mean()) if not rolling_moves_window.empty else np.nan
    sigma_move = float(rolling_moves_window.std()) if not rolling_moves_window.empty else np.nan
    worst_move = float(rolling_moves_window.min()) if not rolling_moves_window.empty else np.nan
    best_move = float(rolling_moves_window.max()) if not rolling_moves_window.empty else np.nan

    snapshot_table = pd.DataFrame(
        [
            {
                "Structure": structure_label,
                "Level": latest_level,
                "Chg vs 1M Avg": change_vs_one_month_avg,
                "1M Avg": one_month_avg,
                "3M Avg": three_month_avg,
                "3M High": three_month_high,
                "3M Low": three_month_low,
                "3M Pct (%)": three_month_pct,
                "Avg 3M Daily Chg": avg_three_month_daily_change,
                "3M Sigma": three_month_sigma,
                "Latest 62d Move": latest_move,
                "Avg 62d Move (252d)": avg_move,
                "1 Sigma 62d (252d)": sigma_move,
                "Worst 62d (252d)": worst_move,
                "Best 62d (252d)": best_move,
            }
        ]
    )

    snapshot_table.attrs["as_of_date"] = latest_timestamp
    snapshot_table.attrs["as_of_label"] = latest_timestamp.strftime("%Y-%m-%d")

    return snapshot_table


def compute_path_from_start(
    price_window: pd.Series,
    change_type: str = "pct_change",
    value_scale: Optional[float] = None,
) -> pd.Series:
    """Convert a price window into a cumulative path relative to its first observation."""
    if price_window.empty:
        raise ValueError("Price window is empty.")

    baseline = float(price_window.iloc[0])

    if change_type == "pct_change":
        applied_scale = 100.0 if value_scale is None else float(value_scale)
        return ((price_window / baseline) - 1.0) * applied_scale

    if change_type == "difference":
        applied_scale = 1.0 if value_scale is None else float(value_scale)
        return (price_window - baseline) * applied_scale

    raise ValueError("change_type must be either 'pct_change' or 'difference'.")


def _infer_profile_value_label(display_unit: Optional[str]) -> str:
    resolved_unit = (display_unit or "").strip().lower()

    if resolved_unit == "%":
        return "% change"

    if resolved_unit == "bp":
        return "bp change"

    if resolved_unit:
        return f"change ({display_unit})"

    return "change"


def build_daily_intra_period_profile_package(
    raw_data: pd.DataFrame,
    asset_label: str = "Asset",
    price_col: str = "Close",
    change_type: str = "pct_change",
    value_scale: Optional[float] = None,
    display_unit: Optional[str] = None,
    recent_year_count: int = 3,
) -> Dict[str, Any]:
    """Build a same-month daily cumulative profile comparing recent years with historical norms."""
    price_series = prepare_price_series(raw_data=raw_data, price_col=price_col)
    latest_date = pd.Timestamp(price_series.index.max())
    target_month = int(latest_date.month)
    target_month_name = calendar.month_abbr[target_month]

    monthly_profiles: dict[int, pd.Series] = {}
    target_month_series = price_series[price_series.index.month == target_month]
    for year, year_series in target_month_series.groupby(target_month_series.index.year):
        year_series = year_series.sort_index()
        if year_series.empty:
            continue
        cumulative_path = compute_path_from_start(
            price_window=year_series,
            change_type=change_type,
            value_scale=value_scale,
        )
        full_month_index = pd.date_range(
            start=f"{int(year)}-{target_month:02d}-01",
            end=f"{int(year)}-{target_month:02d}-{calendar.monthrange(int(year), target_month)[1]:02d}",
            freq="D",
        )
        calendar_day_path = cumulative_path.reindex(full_month_index).ffill().fillna(0.0)

        if int(year) == int(latest_date.year) and target_month == int(latest_date.month):
            calendar_day_path.loc[latest_date + pd.Timedelta(days=1) :] = np.nan

        monthly_profiles[int(year)] = pd.Series(
            calendar_day_path.values,
            index=calendar_day_path.index.day,
            dtype=float,
        )

    if not monthly_profiles:
        raise ValueError("Not enough data to build the daily intra-period profile.")

    month_day_count = max(
        calendar.monthrange(int(year), target_month)[1]
        for year in monthly_profiles
    )
    day_range = list(range(1, month_day_count + 1))
    profile_matrix = pd.DataFrame(monthly_profiles).T.reindex(columns=day_range)
    profile_matrix.index.name = "year"

    benchmark_matrix = profile_matrix.copy()
    average_profile = benchmark_matrix.mean(axis=0)
    median_profile = benchmark_matrix.median(axis=0)

    available_years_desc = sorted(profile_matrix.index.tolist(), reverse=True)
    displayed_years = available_years_desc[: max(1, recent_year_count)]

    value_label = _infer_profile_value_label(display_unit)
    if (display_unit or "").strip() == "%":
        figure_title = f"{asset_label} — {target_month_name} Daily Intra-Period Cum % Change"
        y_axis_title = "Cumulative % change from start of month"
    elif (display_unit or "").strip().lower() == "bp":
        figure_title = f"{asset_label} — {target_month_name} Daily Intra-Period Cumulative Change (bp)"
        y_axis_title = "Cumulative bp change from start of month"
    else:
        figure_title = f"{asset_label} — {target_month_name} Daily Intra-Period Cumulative Change"
        y_axis_title = "Cumulative change from start of month"

    recent_year_styles = [
        ("#e31a1c", "solid", 3.0),
        ("#33a02c", "dot", 2.2),
        ("#3b4cc0", "dot", 2.2),
    ]

    fig = go.Figure()
    for idx, year in enumerate(displayed_years):
        color, dash, width = recent_year_styles[min(idx, len(recent_year_styles) - 1)]
        fig.add_trace(
            go.Scatter(
                x=profile_matrix.columns.tolist(),
                y=profile_matrix.loc[year].values,
                mode="lines",
                name=str(year),
                line={"color": color, "width": width, "dash": dash},
                hovertemplate=(
                    "Day: %{x}<br>"
                    + f"{year}: %{{y:.2f}}"
                    + (display_unit or "")
                    + "<extra></extra>"
                ),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=profile_matrix.columns.tolist(),
            y=average_profile.values,
            mode="lines",
            name="Average",
            line={"color": "#f0ad00", "width": 2.5, "dash": "dash"},
            hovertemplate="Day: %{x}<br>Average: %{y:.2f}" + (display_unit or "") + "<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=profile_matrix.columns.tolist(),
            y=median_profile.values,
            mode="lines",
            name="Median",
            line={"color": "#7f7f7f", "width": 2.5, "dash": "dash"},
            hovertemplate="Day: %{x}<br>Median: %{y:.2f}" + (display_unit or "") + "<extra></extra>",
        )
    )

    fig.add_hline(y=0, line_dash="dot", line_color="rgba(0, 0, 0, 0.28)")
    fig.update_layout(
        title=figure_title,
        template="plotly_white",
        hovermode="x unified",
        height=420,
        margin={"l": 60, "r": 20, "t": 70, "b": 50},
        xaxis_title="Day of Month",
        yaxis_title=y_axis_title,
        legend={"orientation": "v", "yanchor": "top", "y": 0.98, "xanchor": "left", "x": 1.01},
    )

    return {
        "price_series": price_series,
        "profile_matrix": profile_matrix,
        "average_profile": average_profile,
        "median_profile": median_profile,
        "figure": fig,
        "config": {
            "asset_label": asset_label,
            "price_col": price_col,
            "change_type": change_type,
            "value_scale": value_scale,
            "display_unit": display_unit,
            "target_month": target_month,
            "target_month_name": target_month_name,
            "value_label": value_label,
        },
    }


def build_calendar_year_cumulative_profile_package(
    raw_data: pd.DataFrame,
    asset_label: str = "Asset",
    price_col: str = "Close",
    change_type: str = "pct_change",
    value_scale: Optional[float] = None,
    display_unit: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a calendar-year cumulative path chart with historical average and percentile bands."""
    price_series = prepare_price_series(raw_data=raw_data, price_col=price_col)
    latest_date = pd.Timestamp(price_series.index.max()).normalize()
    latest_year = int(latest_date.year)

    reference_dates = pd.date_range("2001-01-01", "2001-12-31", freq="D")
    reference_labels = reference_dates.strftime("%m-%d")
    calendar_paths: dict[int, pd.Series] = {}
    complete_years: list[int] = []

    for year, year_series in price_series.groupby(price_series.index.year):
        year_series = year_series.sort_index()
        if year_series.empty:
            continue

        cumulative_path = compute_path_from_start(
            price_window=year_series,
            change_type=change_type,
            value_scale=value_scale,
        )
        full_calendar_index = pd.date_range(f"{int(year)}-01-01", f"{int(year)}-12-31", freq="D")
        calendar_series = cumulative_path.reindex(full_calendar_index).ffill().fillna(0.0)

        year_end = pd.Timestamp(f"{int(year)}-12-31")
        if int(year) == latest_year and latest_date < year_end:
            calendar_series.loc[latest_date + pd.Timedelta(days=1) :] = np.nan

        if year_series.index.min().month == 1 and year_series.index.max().month == 12:
            complete_years.append(int(year))

        calendar_series = calendar_series[~((calendar_series.index.month == 2) & (calendar_series.index.day == 29))]
        calendar_series.index = calendar_series.index.strftime("%m-%d")
        calendar_paths[int(year)] = calendar_series.reindex(reference_labels)

    if not calendar_paths:
        raise ValueError("Not enough data to build the calendar-year cumulative profile.")

    calendar_matrix = pd.DataFrame(calendar_paths).T.reindex(columns=reference_labels)
    calendar_matrix.index.name = "year"

    benchmark_years = [year for year in complete_years if year != latest_year]
    if benchmark_years:
        benchmark_matrix = calendar_matrix.loc[benchmark_years]
    else:
        benchmark_matrix = calendar_matrix.drop(index=latest_year, errors="ignore")
        if benchmark_matrix.empty:
            benchmark_matrix = calendar_matrix

    average_path = benchmark_matrix.mean(axis=0)
    pct30_path = benchmark_matrix.quantile(0.30, axis=0)
    pct70_path = benchmark_matrix.quantile(0.70, axis=0)
    latest_path = calendar_matrix.loc[latest_year]

    resolved_display_unit = (display_unit or "").strip()
    if resolved_display_unit == "%":
        figure_title = f"{asset_label} — Cumulative % Change over Calendar Year"
        y_axis_title = "Cumulative % change (YTD)"
    elif resolved_display_unit.lower() == "bp":
        figure_title = f"{asset_label} — Cumulative Change over Calendar Year (bp)"
        y_axis_title = "Cumulative change (bp)"
    else:
        figure_title = f"{asset_label} — Cumulative Change over Calendar Year"
        y_axis_title = "Cumulative change (YTD)"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=reference_dates,
            y=latest_path.values,
            mode="lines",
            name=str(latest_year),
            line={"color": "#e31a1c", "width": 3.0},
            hovertemplate="%{x|%b %d}<br>" + f"{latest_year}: %{{y:.2f}}" + resolved_display_unit + "<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=reference_dates,
            y=average_path.values,
            mode="lines",
            name="Average",
            line={"color": "#b39b00", "width": 3.0},
            hovertemplate="%{x|%b %d}<br>Average: %{y:.2f}" + resolved_display_unit + "<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=reference_dates,
            y=pct30_path.values,
            mode="lines",
            name="30th pct",
            line={"color": "#33a02c", "width": 2.4, "dash": "dot"},
            hovertemplate="%{x|%b %d}<br>30th pct: %{y:.2f}" + resolved_display_unit + "<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=reference_dates,
            y=pct70_path.values,
            mode="lines",
            name="70th pct",
            line={"color": "#fb6a4a", "width": 2.4, "dash": "dot"},
            hovertemplate="%{x|%b %d}<br>70th pct: %{y:.2f}" + resolved_display_unit + "<extra></extra>",
        )
    )

    fig.add_hline(y=0, line_dash="dot", line_color="rgba(0, 0, 0, 0.28)")
    fig.update_layout(
        title=figure_title,
        template="plotly_white",
        hovermode="x unified",
        height=420,
        margin={"l": 60, "r": 20, "t": 70, "b": 50},
        xaxis_title="Month (Jan-Dec)",
        yaxis_title=y_axis_title,
        legend={"orientation": "v", "yanchor": "top", "y": 0.98, "xanchor": "left", "x": 1.01},
    )
    fig.update_xaxes(
        tickvals=pd.date_range("2001-01-01", "2001-12-01", freq="MS"),
        tickformat="%b",
    )

    return {
        "price_series": price_series,
        "calendar_matrix": calendar_matrix,
        "average_path": average_path,
        "pct30_path": pct30_path,
        "pct70_path": pct70_path,
        "figure": fig,
        "config": {
            "asset_label": asset_label,
            "price_col": price_col,
            "change_type": change_type,
            "value_scale": value_scale,
            "display_unit": display_unit,
            "latest_year": latest_year,
        },
    }


def build_latest_year_monthly_comparison_package(
    raw_data: pd.DataFrame,
    asset_label: str = "Asset",
    price_col: str = "Close",
    change_type: str = "pct_change",
    value_scale: Optional[float] = None,
    display_unit: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a monthly seasonality comparison chart: historical average/median vs latest year."""
    price_series = prepare_price_series(raw_data=raw_data, price_col=price_col)
    monthly_df_complete = compute_monthly_changes(
        price_series=price_series,
        change_type=change_type,
        drop_incomplete_last_month=True,
        value_scale=value_scale,
    )
    seasonality_matrix, summary_stats = build_monthly_seasonality_tables(monthly_df_complete)

    monthly_df_latest = compute_monthly_changes(
        price_series=price_series,
        change_type=change_type,
        drop_incomplete_last_month=False,
        value_scale=value_scale,
    )
    latest_year = int(monthly_df_latest["year"].max())
    latest_year_matrix = monthly_df_latest.pivot_table(
        index="year",
        columns="month",
        values="monthly_change",
        aggfunc="mean",
    ).reindex(columns=MONTH_NUMBERS)
    latest_year_matrix.columns = MONTH_LABELS
    latest_year_row = latest_year_matrix.loc[latest_year].reindex(MONTH_LABELS)

    x_labels = [
        f"{month_label} ({int(round(win_rate))}%)"
        for month_label, win_rate in zip(summary_stats["month_label"].astype(str), summary_stats["win_rate"])
    ]

    resolved_display_unit = (display_unit or "").strip()
    if resolved_display_unit == "%":
        figure_title = f"{asset_label} — Monthly Seasonality % (Latest Year vs Avg/Median)"
        y_axis_title = "MoM % change"
    elif resolved_display_unit.lower() == "bp":
        figure_title = f"{asset_label} — Monthly Seasonality (Latest Year vs Avg/Median, bp)"
        y_axis_title = "MoM change (bp)"
    else:
        figure_title = f"{asset_label} — Monthly Seasonality (Latest Year vs Avg/Median)"
        y_axis_title = "MoM change"

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=summary_stats["average"],
            name="Average",
            marker_color="#8c8c8c",
            hovertemplate="%{x}<br>Average: %{y:.2f}" + resolved_display_unit + "<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=summary_stats["median"],
            name="Median",
            marker_color="#d9d9d9",
            hovertemplate="%{x}<br>Median: %{y:.2f}" + resolved_display_unit + "<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=latest_year_row.values,
            mode="lines+markers",
            name=str(latest_year),
            line={"color": "#e31a1c", "width": 3.0},
            marker={"size": 8, "color": "#e31a1c"},
            hovertemplate="%{x}<br>" + f"{latest_year}: %{{y:.2f}}" + resolved_display_unit + "<extra></extra>",
        )
    )

    fig.add_hline(y=0, line_dash="dot", line_color="rgba(0, 0, 0, 0.28)")
    fig.update_layout(
        title=figure_title,
        template="plotly_white",
        barmode="group",
        hovermode="x unified",
        height=420,
        margin={"l": 60, "r": 20, "t": 70, "b": 95},
        xaxis_title="Month (Hit Rate %)",
        yaxis_title=y_axis_title,
        legend={"orientation": "v", "yanchor": "top", "y": 0.98, "xanchor": "left", "x": 1.01},
    )
    fig.update_xaxes(tickangle=25)

    return {
        "price_series": price_series,
        "monthly_changes_complete": monthly_df_complete,
        "monthly_changes_latest": monthly_df_latest,
        "seasonality_matrix": seasonality_matrix,
        "summary_stats": summary_stats,
        "latest_year": latest_year,
        "latest_year_row": latest_year_row,
        "figure": fig,
        "config": {
            "asset_label": asset_label,
            "price_col": price_col,
            "change_type": change_type,
            "value_scale": value_scale,
            "display_unit": display_unit,
        },
    }


def compute_monthly_changes(
    price_series: pd.Series,
    change_type: str = "pct_change",
    drop_incomplete_last_month: bool = True,
    value_scale: Optional[float] = None,
) -> pd.DataFrame:
    """Convert a daily price series into month-end changes for seasonality analysis."""
    month_end_prices = price_series.resample("ME").last().dropna()

    if month_end_prices.shape[0] < 2:
        raise ValueError("Not enough month-end observations to calculate monthly seasonality.")

    if drop_incomplete_last_month:
        latest_observation = price_series.index.max().normalize()
        latest_calendar_month_end = latest_observation.to_period("M").to_timestamp("M")

        if (
            not month_end_prices.empty
            and month_end_prices.index.max().normalize() == latest_calendar_month_end
            and latest_observation < latest_calendar_month_end
        ):
            month_end_prices = month_end_prices.iloc[:-1]

    if month_end_prices.shape[0] < 2:
        raise ValueError("Not enough complete month-end observations after filtering incomplete months.")

    if change_type == "pct_change":
        monthly_change = month_end_prices.pct_change().dropna()
        applied_scale = 100.0 if value_scale is None else value_scale
    elif change_type == "difference":
        monthly_change = month_end_prices.diff().dropna()
        applied_scale = 1.0 if value_scale is None else value_scale
    else:
        raise ValueError("change_type must be either 'pct_change' or 'difference'.")

    monthly_change = monthly_change * applied_scale

    monthly_df = monthly_change.to_frame(name="monthly_change").reset_index()
    monthly_df.columns = ["date", "monthly_change"]
    monthly_df["year"] = monthly_df["date"].dt.year
    monthly_df["month"] = monthly_df["date"].dt.month
    monthly_df["month_label"] = pd.Categorical(
        monthly_df["month"].map(lambda month: calendar.month_abbr[month]),
        categories=MONTH_LABELS,
        ordered=True,
    )

    return monthly_df.sort_values(["year", "month"]).reset_index(drop=True)


def build_monthly_seasonality_tables(monthly_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a year-by-month matrix and a monthly summary table."""
    if monthly_df.empty:
        raise ValueError("Monthly change data is empty.")

    seasonality_matrix = monthly_df.pivot_table(
        index="year",
        columns="month",
        values="monthly_change",
        aggfunc="mean",
    ).reindex(columns=MONTH_NUMBERS)
    seasonality_matrix.columns = MONTH_LABELS

    summary_stats = (
        monthly_df.groupby("month", as_index=False)["monthly_change"]
        .agg(
            average="mean",
            median="median",
            volatility="std",
            win_rate=lambda values: (values > 0).mean() * 100,
            observations="count",
        )
        .sort_values("month")
    )
    summary_stats["month_label"] = pd.Categorical(
        summary_stats["month"].map(lambda month: calendar.month_abbr[month]),
        categories=MONTH_LABELS,
        ordered=True,
    )
    summary_stats = summary_stats[
        ["month_label", "average", "median", "volatility", "win_rate", "observations"]
    ]

    return seasonality_matrix, summary_stats


def create_monthly_seasonality_figure(
    seasonality_matrix: pd.DataFrame,
    summary_stats: pd.DataFrame,
    asset_label: str = "Asset",
    y_axis_title: str = "Monthly Return (%)",
    highlight_year: Optional[int] = None,
) -> go.Figure:
    """Create a Streamlit-friendly Plotly figure for monthly seasonality."""
    fig = go.Figure()

    for year, row in seasonality_matrix.iterrows():
        is_highlighted = highlight_year is not None and int(year) == int(highlight_year)
        line_color = "rgba(31, 119, 180, 0.95)" if is_highlighted else "rgba(160, 160, 160, 0.28)"
        line_width = 2.5 if is_highlighted else 1.2
        marker_size = 6 if is_highlighted else 4
        trace_name = f"{year}" if is_highlighted else str(year)

        fig.add_trace(
            go.Scatter(
                x=MONTH_LABELS,
                y=row.values,
                mode="lines+markers",
                name=trace_name,
                showlegend=is_highlighted,
                line={"color": line_color, "width": line_width},
                marker={"size": marker_size, "color": line_color},
                hovertemplate=str(year) + ": %{y:.2f}<extra></extra>",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=summary_stats["month_label"].astype(str),
            y=summary_stats["average"],
            mode="lines+markers",
            name="Average",
            line={"color": "black", "width": 4},
            marker={"size": 8, "color": "black"},
            hovertemplate="Average: %{y:.2f}<extra></extra>",
        )
    )

    fig.add_hline(y=0, line_dash="dot", line_color="rgba(0, 0, 0, 0.35)")
    fig.update_layout(
        title=f"{asset_label} Monthly Seasonality",
        template="plotly_white",
        hovermode="x unified",
        height=520,
        margin={"l": 50, "r": 30, "t": 70, "b": 50},
        xaxis_title="Month",
        yaxis_title=y_axis_title,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    )

    return fig


def build_monthly_seasonality_package(
    raw_data: pd.DataFrame,
    asset_label: str = "Asset",
    price_col: str = "Close",
    change_type: str = "pct_change",
    drop_incomplete_last_month: bool = True,
    highlight_year: Optional[int] = None,
    value_scale: Optional[float] = None,
    y_axis_title: Optional[str] = None,
) -> Dict[str, Any]:
    """End-to-end monthly seasonality pipeline for notebooks and Streamlit."""
    price_series = prepare_price_series(raw_data=raw_data, price_col=price_col)
    monthly_df = compute_monthly_changes(
        price_series=price_series,
        change_type=change_type,
        drop_incomplete_last_month=drop_incomplete_last_month,
        value_scale=value_scale,
    )
    seasonality_matrix, summary_stats = build_monthly_seasonality_tables(monthly_df)

    inferred_y_axis_title = y_axis_title
    if inferred_y_axis_title is None:
        inferred_y_axis_title = "Monthly Return (%)" if change_type == "pct_change" else "Monthly Change"

    figure = create_monthly_seasonality_figure(
        seasonality_matrix=seasonality_matrix,
        summary_stats=summary_stats,
        asset_label=asset_label,
        y_axis_title=inferred_y_axis_title,
        highlight_year=highlight_year,
    )

    return {
        "price_series": price_series,
        "monthly_changes": monthly_df,
        "seasonality_matrix": seasonality_matrix,
        "summary_stats": summary_stats,
        "figure": figure,
        "config": {
            "asset_label": asset_label,
            "price_col": price_col,
            "change_type": change_type,
            "drop_incomplete_last_month": drop_incomplete_last_month,
            "highlight_year": highlight_year,
            "value_scale": value_scale,
            "y_axis_title": inferred_y_axis_title,
        },
    }


def compute_cumulative_path(
    monthly_matrix: pd.DataFrame,
    change_type: str = "pct_change",
    value_scale: Optional[float] = None,
) -> pd.DataFrame:
    """Convert a year-by-month matrix of monthly changes into cumulative intra-year paths."""
    if monthly_matrix.empty:
        raise ValueError("Monthly matrix is empty. Cannot compute cumulative seasonality.")

    cumulative_matrix = monthly_matrix.copy()

    if change_type == "pct_change":
        divisor = 100.0 if value_scale is None else value_scale
        cumulative_matrix = ((1 + cumulative_matrix / divisor).cumprod(axis=1) - 1) * divisor
    elif change_type == "difference":
        cumulative_matrix = cumulative_matrix.cumsum(axis=1)
    else:
        raise ValueError("change_type must be either 'pct_change' or 'difference'.")

    return cumulative_matrix


def build_cumulative_seasonality_tables(
    monthly_df: pd.DataFrame,
    change_type: str = "pct_change",
    value_scale: Optional[float] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return monthly and cumulative seasonality tables plus summary statistics."""
    if monthly_df.empty:
        raise ValueError("Monthly change data is empty.")

    monthly_matrix = monthly_df.pivot_table(
        index="year",
        columns="month",
        values="monthly_change",
        aggfunc="mean",
    ).reindex(columns=MONTH_NUMBERS)
    monthly_matrix.columns = MONTH_LABELS

    cumulative_matrix = compute_cumulative_path(
        monthly_matrix=monthly_matrix,
        change_type=change_type,
        value_scale=value_scale,
    )

    summary_stats = pd.DataFrame(
        {
            "month_label": MONTH_LABELS,
            "average": cumulative_matrix.mean(axis=0).reindex(MONTH_LABELS).values,
            "median": cumulative_matrix.median(axis=0).reindex(MONTH_LABELS).values,
            "volatility": cumulative_matrix.std(axis=0).reindex(MONTH_LABELS).values,
            "positive_rate": ((cumulative_matrix > 0).mean(axis=0) * 100).reindex(MONTH_LABELS).values,
            "observations": cumulative_matrix.count(axis=0).reindex(MONTH_LABELS).values,
        }
    )

    return monthly_matrix, cumulative_matrix, summary_stats


def create_cumulative_seasonality_figure(
    cumulative_matrix: pd.DataFrame,
    summary_stats: pd.DataFrame,
    asset_label: str = "Asset",
    y_axis_title: str = "Cumulative Return (%)",
    highlight_year: Optional[int] = None,
) -> go.Figure:
    """Create a Streamlit-friendly Plotly figure for cumulative seasonality."""
    fig = go.Figure()

    for year, row in cumulative_matrix.iterrows():
        is_highlighted = highlight_year is not None and int(year) == int(highlight_year)
        line_color = "rgba(31, 119, 180, 0.95)" if is_highlighted else "rgba(160, 160, 160, 0.28)"
        line_width = 2.5 if is_highlighted else 1.2
        marker_size = 6 if is_highlighted else 4
        trace_name = f"{year}" if is_highlighted else str(year)

        fig.add_trace(
            go.Scatter(
                x=MONTH_LABELS,
                y=row.values,
                mode="lines+markers",
                name=trace_name,
                showlegend=is_highlighted,
                line={"color": line_color, "width": line_width},
                marker={"size": marker_size, "color": line_color},
                hovertemplate=str(year) + ": %{y:.2f}<extra></extra>",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=summary_stats["month_label"].astype(str),
            y=summary_stats["average"],
            mode="lines+markers",
            name="Average",
            line={"color": "black", "width": 4},
            marker={"size": 8, "color": "black"},
            hovertemplate="Average: %{y:.2f}<extra></extra>",
        )
    )

    fig.add_hline(y=0, line_dash="dot", line_color="rgba(0, 0, 0, 0.35)")
    fig.update_layout(
        title=f"{asset_label} Cumulative Seasonality",
        template="plotly_white",
        hovermode="x unified",
        height=520,
        margin={"l": 50, "r": 30, "t": 70, "b": 50},
        xaxis_title="Month",
        yaxis_title=y_axis_title,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    )

    return fig


def build_cumulative_seasonality_package(
    raw_data: pd.DataFrame,
    asset_label: str = "Asset",
    price_col: str = "Close",
    change_type: str = "pct_change",
    drop_incomplete_last_month: bool = True,
    highlight_year: Optional[int] = None,
    value_scale: Optional[float] = None,
    y_axis_title: Optional[str] = None,
) -> Dict[str, Any]:
    """End-to-end cumulative seasonality pipeline for notebooks and Streamlit."""
    price_series = prepare_price_series(raw_data=raw_data, price_col=price_col)
    monthly_df = compute_monthly_changes(
        price_series=price_series,
        change_type=change_type,
        drop_incomplete_last_month=drop_incomplete_last_month,
        value_scale=value_scale,
    )
    monthly_matrix, cumulative_matrix, summary_stats = build_cumulative_seasonality_tables(
        monthly_df=monthly_df,
        change_type=change_type,
        value_scale=value_scale,
    )

    inferred_y_axis_title = y_axis_title
    if inferred_y_axis_title is None:
        inferred_y_axis_title = "Cumulative Return (%)" if change_type == "pct_change" else "Cumulative Change"

    figure = create_cumulative_seasonality_figure(
        cumulative_matrix=cumulative_matrix,
        summary_stats=summary_stats,
        asset_label=asset_label,
        y_axis_title=inferred_y_axis_title,
        highlight_year=highlight_year,
    )

    return {
        "price_series": price_series,
        "monthly_changes": monthly_df,
        "monthly_matrix": monthly_matrix,
        "cumulative_matrix": cumulative_matrix,
        "summary_stats": summary_stats,
        "figure": figure,
        "config": {
            "asset_label": asset_label,
            "price_col": price_col,
            "change_type": change_type,
            "drop_incomplete_last_month": drop_incomplete_last_month,
            "highlight_year": highlight_year,
            "value_scale": value_scale,
            "y_axis_title": inferred_y_axis_title,
        },
    }


def build_heatmap_matrix(
    monthly_df: pd.DataFrame,
    sort_descending: bool = True,
    demean_columns: bool = False,
) -> pd.DataFrame:
    """Return a year-by-month matrix for seasonality heatmap visualization."""
    if monthly_df.empty:
        raise ValueError("Monthly change data is empty.")

    heatmap_matrix = monthly_df.pivot_table(
        index="year",
        columns="month",
        values="monthly_change",
        aggfunc="mean",
    ).reindex(columns=MONTH_NUMBERS)
    heatmap_matrix.columns = MONTH_LABELS

    if demean_columns:
        heatmap_matrix = heatmap_matrix.sub(heatmap_matrix.mean(axis=0, skipna=True), axis=1)

    heatmap_matrix = heatmap_matrix.sort_index(ascending=not sort_descending)

    return heatmap_matrix


def infer_display_unit(
    change_type: str,
    display_unit: Optional[str] = None,
) -> str:
    """Infer the displayed unit: bonds typically use bp, most other assets use percent."""
    if display_unit is not None:
        return display_unit

    return "%" if change_type == "pct_change" else "bp"


def infer_value_label(
    change_type: str,
    display_unit: str,
) -> str:
    """Build a human-readable value label for tables and hover output."""
    if display_unit == "%":
        return "Monthly Return (%)"

    if display_unit.lower() == "bp":
        return "Monthly Change (bp)"

    return "Monthly Change"


def compute_heatmap_clip_limits(
    heatmap_matrix: pd.DataFrame,
    clip_quantile: float = 0.95,
    minimum_clip_value: float = 0.5,
    clip_mode: str = "row",
) -> pd.Series:
    """Compute clipping limits for each row or for the whole heatmap."""
    if heatmap_matrix.empty:
        raise ValueError("Heatmap matrix is empty.")

    if clip_mode not in {"row", "global"}:
        raise ValueError("clip_mode must be either 'row' or 'global'.")

    if clip_mode == "global":
        values = heatmap_matrix.to_numpy(dtype=float).ravel()
        values = values[~np.isnan(values)]

        if values.size == 0:
            raise ValueError("Heatmap matrix does not contain any numeric values.")

        robust_range = float(np.quantile(np.abs(values), clip_quantile))
        max_range = float(np.max(np.abs(values)))
        clip_value = robust_range if robust_range > 0 else max_range
        clip_value = max(clip_value, minimum_clip_value)

        return pd.Series(clip_value, index=heatmap_matrix.index, name="clip_limit")

    clip_limits = {}
    for row_index, row_values in heatmap_matrix.iterrows():
        clean_values = row_values.dropna().astype(float).to_numpy()

        if clean_values.size == 0:
            clip_limits[row_index] = minimum_clip_value
            continue

        robust_range = float(np.quantile(np.abs(clean_values), clip_quantile))
        max_range = float(np.max(np.abs(clean_values)))
        clip_value = robust_range if robust_range > 0 else max_range
        clip_limits[row_index] = max(clip_value, minimum_clip_value)

    return pd.Series(clip_limits, name="clip_limit")


def clip_heatmap_values(
    heatmap_matrix: pd.DataFrame,
    clip_limits: pd.Series,
) -> pd.DataFrame:
    """Clip heatmap values row by row to reduce outlier impact on colors."""
    clipped_matrix = heatmap_matrix.copy()

    for row_index in clipped_matrix.index:
        clip_value = float(clip_limits.loc[row_index])
        clipped_matrix.loc[row_index] = clipped_matrix.loc[row_index].clip(
            lower=-clip_value,
            upper=clip_value,
        )

    return clipped_matrix


def normalize_heatmap_for_colors(
    clipped_matrix: pd.DataFrame,
    clip_limits: pd.Series,
) -> pd.DataFrame:
    """Normalize clipped values to [-1, 1] so colors are coded horizontally by row."""
    safe_limits = clip_limits.replace(0, np.nan)
    normalized_matrix = clipped_matrix.div(safe_limits, axis=0)

    return normalized_matrix.clip(lower=-1, upper=1)


def format_heatmap_text(
    value_matrix: pd.DataFrame,
    display_unit: str,
    decimals: int = 1,
) -> pd.DataFrame:
    """Format heatmap values for in-cell display."""
    return value_matrix.apply(
        lambda column: column.map(
            lambda value: "" if pd.isna(value) else f"{value:.{decimals}f}{display_unit}"
        )
    )


def create_seasonality_heatmap_figure(
    heatmap_matrix: pd.DataFrame,
    asset_label: str = "Asset",
    change_type: str = "pct_change",
    display_unit: Optional[str] = None,
    value_label: Optional[str] = None,
    clip_quantile: float = 0.95,
    minimum_clip_value: float = 0.5,
    clip_mode: str = "row",
    reverse_y_axis: bool = True,
    decimals: int = 1,
) -> go.Figure:
    """Create a red/green heatmap with row-wise color coding and clipped outliers."""
    if heatmap_matrix.empty:
        raise ValueError("Heatmap matrix is empty.")

    resolved_display_unit = infer_display_unit(
        change_type=change_type,
        display_unit=display_unit,
    )
    resolved_value_label = value_label or infer_value_label(
        change_type=change_type,
        display_unit=resolved_display_unit,
    )

    clip_limits = compute_heatmap_clip_limits(
        heatmap_matrix=heatmap_matrix,
        clip_quantile=clip_quantile,
        minimum_clip_value=minimum_clip_value,
        clip_mode=clip_mode,
    )
    clipped_matrix = clip_heatmap_values(
        heatmap_matrix=heatmap_matrix,
        clip_limits=clip_limits,
    )
    color_matrix = normalize_heatmap_for_colors(
        clipped_matrix=clipped_matrix,
        clip_limits=clip_limits,
    )
    text_matrix = format_heatmap_text(
        value_matrix=heatmap_matrix,
        display_unit=resolved_display_unit,
        decimals=decimals,
    )

    colorscale = [
        [0.00, "#d73027"],
        [0.50, "#ffffff"],
        [1.00, "#1a9850"],
    ]

    hover_value_format = f"%{{customdata:.{decimals}f}}{resolved_display_unit}"
    figure_height = max(420, 28 * len(heatmap_matrix.index) + 140)

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=color_matrix.values,
                x=heatmap_matrix.columns.tolist(),
                y=heatmap_matrix.index.astype(str).tolist(),
                customdata=heatmap_matrix.values,
                text=text_matrix.values,
                texttemplate="%{text}",
                colorscale=colorscale,
                zmid=0,
                zmin=-1,
                zmax=1,
                colorbar={
                    "title": "Row Color Scale",
                    "tickvals": [-1, 0, 1],
                    "ticktext": ["Negative", "Flat", "Positive"],
                },
                xgap=1,
                ygap=1,
                hovertemplate=(
                    "Year: %{y}<br>"
                    + "Month: %{x}<br>"
                    + resolved_value_label
                    + ": "
                    + hover_value_format
                    + "<extra></extra>"
                ),
            )
        ]
    )

    fig.update_layout(
        title=f"{asset_label} Seasonality Heatmap",
        template="plotly_white",
        height=figure_height,
        margin={"l": 60, "r": 40, "t": 70, "b": 50},
        xaxis_title="Month",
        yaxis_title="Year",
    )

    if reverse_y_axis:
        fig.update_yaxes(autorange="reversed")

    return fig


def build_seasonality_heatmap_package(
    raw_data: pd.DataFrame,
    asset_label: str = "Asset",
    price_col: str = "Close",
    change_type: str = "pct_change",
    drop_incomplete_last_month: bool = True,
    value_scale: Optional[float] = None,
    display_unit: Optional[str] = None,
    value_label: Optional[str] = None,
    clip_quantile: float = 0.95,
    minimum_clip_value: float = 0.5,
    clip_mode: str = "row",
    sort_descending: bool = True,
    decimals: int = 1,
    demean_columns: bool = False,
) -> Dict[str, Any]:
    """End-to-end heatmap pipeline for notebooks and Streamlit."""
    price_series = prepare_price_series(raw_data=raw_data, price_col=price_col)
    monthly_df = compute_monthly_changes(
        price_series=price_series,
        change_type=change_type,
        drop_incomplete_last_month=drop_incomplete_last_month,
        value_scale=value_scale,
    )
    heatmap_matrix = build_heatmap_matrix(
        monthly_df=monthly_df,
        sort_descending=sort_descending,
        demean_columns=demean_columns,
    )

    resolved_display_unit = infer_display_unit(
        change_type=change_type,
        display_unit=display_unit,
    )
    resolved_value_label = value_label or infer_value_label(
        change_type=change_type,
        display_unit=resolved_display_unit,
    )
    clip_limits = compute_heatmap_clip_limits(
        heatmap_matrix=heatmap_matrix,
        clip_quantile=clip_quantile,
        minimum_clip_value=minimum_clip_value,
        clip_mode=clip_mode,
    )
    clipped_matrix = clip_heatmap_values(
        heatmap_matrix=heatmap_matrix,
        clip_limits=clip_limits,
    )
    color_matrix = normalize_heatmap_for_colors(
        clipped_matrix=clipped_matrix,
        clip_limits=clip_limits,
    )

    figure = create_seasonality_heatmap_figure(
        heatmap_matrix=heatmap_matrix,
        asset_label=asset_label,
        change_type=change_type,
        display_unit=resolved_display_unit,
        value_label=resolved_value_label,
        clip_quantile=clip_quantile,
        minimum_clip_value=minimum_clip_value,
        clip_mode=clip_mode,
        reverse_y_axis=True,
        decimals=decimals,
    )

    return {
        "price_series": price_series,
        "monthly_changes": monthly_df,
        "heatmap_matrix": heatmap_matrix,
        "clipped_matrix": clipped_matrix,
        "color_matrix": color_matrix,
        "clip_limits": clip_limits,
        "figure": figure,
        "config": {
            "asset_label": asset_label,
            "price_col": price_col,
            "change_type": change_type,
            "drop_incomplete_last_month": drop_incomplete_last_month,
            "value_scale": value_scale,
            "display_unit": resolved_display_unit,
            "demean_columns": demean_columns,
            "value_label": resolved_value_label,
            "clip_quantile": clip_quantile,
            "minimum_clip_value": minimum_clip_value,
            "clip_mode": clip_mode,
            "sort_descending": sort_descending,
            "decimals": decimals,
        },
    }


def normalize_event_dates(event_dates) -> pd.DatetimeIndex:
    """Parse, clean, and sort a user-supplied event date list."""
    if isinstance(event_dates, (str, datetime, pd.Timestamp)):
        raw_values = [event_dates]
    else:
        raw_values = list(event_dates)

    if not raw_values:
        raise ValueError("event_dates must contain at least one valid date.")

    parsed = pd.to_datetime(pd.Series(raw_values), errors="coerce").dropna()

    if parsed.empty:
        raise ValueError("No valid event dates were supplied.")

    if parsed.dt.tz is not None:
        parsed = parsed.dt.tz_localize(None)

    parsed = parsed.dt.normalize().drop_duplicates().sort_values().reset_index(drop=True)
    return pd.DatetimeIndex(parsed.tolist())


def infer_event_value_scale(
    change_type: str,
    display_unit: str,
    value_scale: Optional[float] = None,
) -> float:
    """Infer the scale used for event-study values."""
    if value_scale is not None:
        return float(value_scale)

    if change_type == "pct_change":
        return 100.0

    if display_unit.lower() == "bp":
        return 100.0

    return 1.0


def infer_event_axis_label(display_unit: str) -> str:
    """Build the y-axis label for the event-study chart."""
    if display_unit == "%":
        return "Relative Move (%)"

    if display_unit.lower() == "bp":
        return "Relative Change (bp)"

    return "Relative Change"


def align_event_dates_to_calendar(
    price_series: pd.Series,
    event_dates,
    alignment: str = "next",
) -> pd.DataFrame:
    """Align event dates to the available trading calendar."""
    if alignment not in {"next", "previous", "nearest"}:
        raise ValueError("alignment must be one of: 'next', 'previous', 'nearest'.")

    normalized_event_dates = normalize_event_dates(event_dates)
    trading_index = pd.DatetimeIndex(price_series.index)

    records = []
    for event_date in normalized_event_dates:
        aligned_position = None

        if event_date in trading_index:
            aligned_position = int(trading_index.get_loc(event_date))
        elif alignment == "next":
            candidate = int(trading_index.searchsorted(event_date, side="left"))
            if candidate < len(trading_index):
                aligned_position = candidate
        elif alignment == "previous":
            candidate = int(trading_index.searchsorted(event_date, side="right")) - 1
            if candidate >= 0:
                aligned_position = candidate
        else:
            right_candidate = int(trading_index.searchsorted(event_date, side="left"))
            left_candidate = right_candidate - 1

            candidate_positions = []
            if left_candidate >= 0:
                candidate_positions.append(left_candidate)
            if right_candidate < len(trading_index):
                candidate_positions.append(right_candidate)

            if candidate_positions:
                aligned_position = min(
                    candidate_positions,
                    key=lambda position: abs((trading_index[position] - event_date).days),
                )

        aligned_event_date = pd.NaT if aligned_position is None else trading_index[aligned_position]
        offset_days = np.nan if pd.isna(aligned_event_date) else int((aligned_event_date - event_date).days)

        records.append(
            {
                "input_event_date": event_date,
                "aligned_event_date": aligned_event_date,
                "aligned_position": aligned_position,
                "calendar_offset_days": offset_days,
                "is_exact_match": bool(not pd.isna(aligned_event_date) and aligned_event_date == event_date),
            }
        )

    aligned_events = pd.DataFrame(records)
    aligned_events = aligned_events.dropna(subset=["aligned_event_date"]).copy()

    if aligned_events.empty:
        raise ValueError("None of the supplied event dates could be aligned to the trading calendar.")

    aligned_events = aligned_events.sort_values("aligned_event_date")
    aligned_events = aligned_events.drop_duplicates(subset=["aligned_event_date"], keep="first")
    aligned_events["event_id"] = aligned_events["aligned_event_date"].dt.strftime("%Y-%m-%d")
    aligned_events["is_current_event"] = False
    aligned_events.loc[aligned_events["aligned_event_date"].idxmax(), "is_current_event"] = True

    return aligned_events.reset_index(drop=True)


def extract_event_window(
    price_series: pd.Series,
    event_position: int,
    window_size: int,
    require_complete_window: bool = True,
) -> Optional[pd.Series]:
    """Extract a trading-day event window centered on the aligned event date."""
    relative_days = pd.Index(range(-window_size, window_size + 1), name="relative_day")
    window_values = pd.Series(np.nan, index=relative_days, dtype=float, name="price")

    available_start = max(-window_size, -event_position)
    available_end = min(window_size, len(price_series) - event_position - 1)

    source_values = price_series.iloc[
        event_position + available_start : event_position + available_end + 1
    ].to_numpy()
    window_values.loc[available_start:available_end] = source_values

    if require_complete_window and window_values.isna().any():
        return None

    return window_values


def compute_event_relative_path(
    window_values: pd.Series,
    change_type: str = "pct_change",
    value_scale: float = 100.0,
) -> pd.Series:
    """Convert a window of prices or yields into a relative event-study path."""
    if 0 not in window_values.index or pd.isna(window_values.loc[0]):
        raise ValueError("Event window does not contain a valid event-day observation.")

    event_value = float(window_values.loc[0])
    relative_path = pd.Series(index=window_values.index, dtype=float, name="relative_value")

    for relative_day, value in window_values.items():
        if pd.isna(value):
            relative_path.loc[relative_day] = np.nan
        elif relative_day == 0:
            relative_path.loc[relative_day] = 0.0
        elif change_type == "pct_change":
            if relative_day < 0:
                relative_path.loc[relative_day] = ((event_value / float(value)) - 1.0) * value_scale
            else:
                relative_path.loc[relative_day] = ((float(value) / event_value) - 1.0) * value_scale
        elif change_type == "difference":
            if relative_day < 0:
                relative_path.loc[relative_day] = (event_value - float(value)) * value_scale
            else:
                relative_path.loc[relative_day] = (float(value) - event_value) * value_scale
        else:
            raise ValueError("change_type must be either 'pct_change' or 'difference'.")

    return relative_path


def build_event_study_paths(
    price_series: pd.Series,
    event_dates,
    window_size: int = 10,
    change_type: str = "pct_change",
    value_scale: float = 100.0,
    alignment: str = "next",
    require_complete_window: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build an event-path matrix and aligned-event metadata table."""
    aligned_events = align_event_dates_to_calendar(
        price_series=price_series,
        event_dates=event_dates,
        alignment=alignment,
    )

    path_records = []
    included_rows = []
    skipped_rows = []

    for _, row in aligned_events.iterrows():
        event_position = int(row["aligned_position"])
        window_values = extract_event_window(
            price_series=price_series,
            event_position=event_position,
            window_size=window_size,
            require_complete_window=require_complete_window,
        )

        if window_values is None:
            skipped_rows.append(
                {
                    **row.to_dict(),
                    "included": False,
                    "skip_reason": "Incomplete event window",
                }
            )
            continue

        relative_path = compute_event_relative_path(
            window_values=window_values,
            change_type=change_type,
            value_scale=value_scale,
        )
        relative_path.name = row["event_id"]
        path_records.append(relative_path)
        included_rows.append(
            {
                **row.to_dict(),
                "included": True,
                "skip_reason": "",
            }
        )

    if not path_records:
        raise ValueError("No valid event windows were created. Check event dates and window length.")

    event_path_matrix = pd.DataFrame(path_records)
    event_path_matrix.index.name = "event_id"
    event_path_matrix.columns.name = "relative_day"

    event_metadata = pd.DataFrame(included_rows + skipped_rows)
    event_metadata = event_metadata.sort_values(["included", "aligned_event_date"], ascending=[False, True])
    event_metadata = event_metadata.reset_index(drop=True)

    return event_path_matrix, event_metadata


def build_event_summary_stats(event_path_matrix: pd.DataFrame) -> pd.DataFrame:
    """Summarize event-study paths by relative day."""
    if event_path_matrix.empty:
        raise ValueError("Event path matrix is empty.")

    summary_stats = pd.DataFrame(
        {
            "relative_day": event_path_matrix.columns.astype(int),
            "average": event_path_matrix.mean(axis=0).values,
            "median": event_path_matrix.median(axis=0).values,
            "volatility": event_path_matrix.std(axis=0).values,
            "positive_rate": ((event_path_matrix > 0).mean(axis=0) * 100).values,
            "observations": event_path_matrix.count(axis=0).values,
        }
    )

    return summary_stats


def format_relative_day_label(relative_day: int) -> str:
    """Format relative trading days as T-10 ... T=0 ... T+10."""
    if relative_day == 0:
        return "T=0"

    return f"T{relative_day:+d}"


def build_event_display_table(
    event_path_matrix: pd.DataFrame,
    event_metadata: pd.DataFrame,
    current_label: str = "Current Event",
    mean_label: str = "Historical Mean",
    sort_descending: bool = True,
) -> pd.DataFrame:
    """Build a display-friendly event table with mean and current-event rows."""
    included_metadata = event_metadata[event_metadata["included"]].copy()
    if included_metadata.empty:
        raise ValueError("No included events are available for the event table.")

    included_metadata = included_metadata.sort_values(
        "aligned_event_date",
        ascending=not sort_descending,
    )
    ordered_event_ids = included_metadata["event_id"].tolist()

    current_event_ids = included_metadata.loc[included_metadata["is_current_event"], "event_id"].tolist()
    current_event_id = current_event_ids[-1] if current_event_ids else ordered_event_ids[0]
    historical_ids = [event_id for event_id in ordered_event_ids if event_id != current_event_id]
    historical_source = event_path_matrix.loc[historical_ids] if historical_ids else event_path_matrix

    display_table = event_path_matrix.loc[ordered_event_ids].copy()
    current_display_label = f"{current_label} | {current_event_id}"
    display_table = display_table.rename(index={current_event_id: current_display_label})

    mean_row = pd.DataFrame([historical_source.mean(axis=0)], index=[mean_label])
    display_table = pd.concat([mean_row, display_table], axis=0)
    display_table.columns = [format_relative_day_label(int(relative_day)) for relative_day in display_table.columns]
    display_table.index.name = "event"

    return display_table


def normalized_value_to_css(normalized_value: float) -> str:
    """Convert a normalized value in [-1, 1] to a red-white-green background color."""
    if pd.isna(normalized_value):
        return "background-color: #f5f5f5; color: #999999;"

    negative_color = np.array([215, 48, 39], dtype=float)
    neutral_color = np.array([255, 255, 255], dtype=float)
    positive_color = np.array([26, 152, 80], dtype=float)

    if normalized_value < 0:
        weight = normalized_value + 1.0
        rgb = negative_color * (1.0 - weight) + neutral_color * weight
    else:
        weight = normalized_value
        rgb = neutral_color * (1.0 - weight) + positive_color * weight

    red, green, blue = rgb.round().astype(int)
    return (
        f"background-color: rgb({red}, {green}, {blue}); "
        "color: #111111; text-align: center;"
    )


def style_event_study_table(
    event_display_table: pd.DataFrame,
    display_unit: str,
    decimals: int = 1,
    clip_quantile: float = 0.95,
    minimum_clip_value: float = 0.5,
    clip_mode: str = "row",
):
    """Style the custom event table using the same red/green row-wise logic as the seasonality heatmap."""
    clip_limits = compute_heatmap_clip_limits(
        heatmap_matrix=event_display_table,
        clip_quantile=clip_quantile,
        minimum_clip_value=minimum_clip_value,
        clip_mode=clip_mode,
    )
    clipped_matrix = clip_heatmap_values(
        heatmap_matrix=event_display_table,
        clip_limits=clip_limits,
    )
    color_matrix = normalize_heatmap_for_colors(
        clipped_matrix=clipped_matrix,
        clip_limits=clip_limits,
    )
    css_matrix = color_matrix.apply(
        lambda column: column.map(normalized_value_to_css)
    )

    styler = event_display_table.style.apply(lambda _: css_matrix, axis=None).format(
        lambda value: "" if pd.isna(value) else f"{value:.{decimals}f}{display_unit}"
    )
    styler = styler.set_table_styles(
        [
            {"selector": "th", "props": [("text-align", "center")]},
            {"selector": "td", "props": [("text-align", "center")]},
        ]
    )

    return styler


def create_custom_event_heatmap_figure(
    event_display_table: pd.DataFrame,
    asset_label: str = "Asset",
    display_unit: str = "%",
    title_label: Optional[str] = None,
    clip_quantile: float = 0.95,
    minimum_clip_value: float = 0.5,
    clip_mode: str = "row",
    reverse_y_axis: bool = True,
    decimals: int = 1,
) -> go.Figure:
    """Render the custom event table as a heatmap image using the same logic as the seasonality heatmap."""
    if event_display_table.empty:
        raise ValueError("Event display table is empty.")

    clip_limits = compute_heatmap_clip_limits(
        heatmap_matrix=event_display_table,
        clip_quantile=clip_quantile,
        minimum_clip_value=minimum_clip_value,
        clip_mode=clip_mode,
    )
    clipped_matrix = clip_heatmap_values(
        heatmap_matrix=event_display_table,
        clip_limits=clip_limits,
    )
    color_matrix = normalize_heatmap_for_colors(
        clipped_matrix=clipped_matrix,
        clip_limits=clip_limits,
    )
    text_matrix = format_heatmap_text(
        value_matrix=event_display_table,
        display_unit=display_unit,
        decimals=decimals,
    )

    colorscale = [
        [0.00, "#d73027"],
        [0.50, "#ffffff"],
        [1.00, "#1a9850"],
    ]

    hover_value_format = f"%{{customdata:.{decimals}f}}{display_unit}"
    figure_height = max(420, 28 * len(event_display_table.index) + 140)
    resolved_title = (
        f"{title_label} | Event Heatmap"
        if title_label
        else f"{asset_label} Custom Event Heatmap"
    )

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=color_matrix.values,
                x=event_display_table.columns.tolist(),
                y=event_display_table.index.tolist(),
                customdata=event_display_table.values,
                text=text_matrix.values,
                texttemplate="%{text}",
                colorscale=colorscale,
                zmid=0,
                zmin=-1,
                zmax=1,
                colorbar={
                    "title": "Row Color Scale",
                    "tickvals": [-1, 0, 1],
                    "ticktext": ["Negative", "Flat", "Positive"],
                },
                xgap=1,
                ygap=1,
                hovertemplate=(
                    "Event: %{y}<br>"
                    + "Relative time: %{x}<br>"
                    + "Value: "
                    + hover_value_format
                    + "<extra></extra>"
                ),
            )
        ]
    )

    fig.update_layout(
        title=resolved_title,
        template="plotly_white",
        height=figure_height,
        margin={"l": 60, "r": 40, "t": 70, "b": 50},
        xaxis_title="Relative Time",
        yaxis_title="Event",
    )

    if reverse_y_axis:
        fig.update_yaxes(autorange="reversed")

    return fig


def create_custom_event_seasonality_figure(
    event_path_matrix: pd.DataFrame,
    event_metadata: pd.DataFrame,
    asset_label: str = "Asset",
    title_label: Optional[str] = None,
    y_axis_title: str = "Relative Move (%)",
    current_label: str = "Current Event",
    mean_label: str = "Historical Mean",
) -> go.Figure:
    """Create an event-study figure with faint historical lines, mean, and current event."""
    if event_path_matrix.empty:
        raise ValueError("Event path matrix is empty.")

    included_metadata = event_metadata[event_metadata["included"]].copy()
    current_event_ids = included_metadata.loc[included_metadata["is_current_event"], "event_id"].tolist()
    current_event_id = current_event_ids[-1] if current_event_ids else event_path_matrix.index[-1]

    historical_ids = [event_id for event_id in event_path_matrix.index if event_id != current_event_id]
    historical_matrix = event_path_matrix.loc[historical_ids] if historical_ids else event_path_matrix.iloc[0:0]
    mean_source = historical_matrix if not historical_matrix.empty else event_path_matrix
    mean_path = mean_source.mean(axis=0)
    current_path = event_path_matrix.loc[current_event_id]

    fig = go.Figure()

    for event_id, row in historical_matrix.iterrows():
        fig.add_trace(
            go.Scatter(
                x=row.index.astype(int).tolist(),
                y=row.values,
                mode="lines",
                name=str(event_id),
                showlegend=False,
                line={"color": "rgba(160, 160, 160, 0.28)", "width": 1.2},
                hovertemplate=str(event_id) + ": %{y:.2f}<extra></extra>",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=mean_path.index.astype(int).tolist(),
            y=mean_path.values,
            mode="lines+markers",
            name=mean_label,
            line={"color": "black", "width": 4},
            marker={"size": 7, "color": "black"},
            hovertemplate=mean_label + ": %{y:.2f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=current_path.index.astype(int).tolist(),
            y=current_path.values,
            mode="lines+markers",
            name=current_label,
            line={"color": "rgba(31, 119, 180, 0.95)", "width": 3},
            marker={"size": 7, "color": "rgba(31, 119, 180, 0.95)"},
            hovertemplate=current_label + ": %{y:.2f}<extra></extra>",
        )
    )

    tick_values = mean_path.index.astype(int).tolist()
    tick_text = [format_relative_day_label(int(value)) for value in tick_values]

    fig.add_vline(x=0, line_dash="dot", line_color="rgba(0, 0, 0, 0.4)")
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(0, 0, 0, 0.25)")
    fig.update_layout(
        title=title_label or f"{asset_label} Custom Event Seasonality",
        template="plotly_white",
        hovermode="x unified",
        height=520,
        margin={"l": 50, "r": 30, "t": 70, "b": 50},
        xaxis_title="Relative Time",
        yaxis_title=y_axis_title,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    )
    fig.update_xaxes(tickmode="array", tickvals=tick_values, ticktext=tick_text)

    return fig


def build_custom_event_seasonality_package(
    raw_data: pd.DataFrame,
    event_dates,
    asset_label: str = "Asset",
    price_col: str = "Close",
    change_type: str = "pct_change",
    window_size: int = 10,
    alignment: str = "next",
    require_complete_window: bool = True,
    value_scale: Optional[float] = None,
    display_unit: Optional[str] = None,
    title_label: Optional[str] = None,
    current_label: str = "Current Event",
    table_decimals: int = 1,
    table_clip_quantile: float = 0.95,
    table_minimum_clip_value: float = 0.5,
    table_clip_mode: str = "row",
) -> Dict[str, Any]:
    """End-to-end custom event seasonality pipeline for notebooks and Streamlit."""
    price_series = prepare_price_series(raw_data=raw_data, price_col=price_col)
    resolved_display_unit = infer_display_unit(change_type=change_type, display_unit=display_unit)
    resolved_value_scale = infer_event_value_scale(
        change_type=change_type,
        display_unit=resolved_display_unit,
        value_scale=value_scale,
    )
    y_axis_title = infer_event_axis_label(display_unit=resolved_display_unit)

    event_path_matrix, event_metadata = build_event_study_paths(
        price_series=price_series,
        event_dates=event_dates,
        window_size=window_size,
        change_type=change_type,
        value_scale=resolved_value_scale,
        alignment=alignment,
        require_complete_window=require_complete_window,
    )

    included_ids = event_metadata.loc[event_metadata["included"], "event_id"].tolist()
    current_ids = event_metadata.loc[
        event_metadata["included"] & event_metadata["is_current_event"],
        "event_id",
    ].tolist()
    current_event_id = current_ids[-1] if current_ids else included_ids[-1]
    historical_ids = [event_id for event_id in included_ids if event_id != current_event_id]

    historical_matrix = event_path_matrix.loc[historical_ids] if historical_ids else event_path_matrix.copy()
    historical_mean_path = historical_matrix.mean(axis=0)
    current_event_path = event_path_matrix.loc[current_event_id]
    summary_stats = build_event_summary_stats(historical_matrix)
    event_display_table = build_event_display_table(
        event_path_matrix=event_path_matrix,
        event_metadata=event_metadata,
        current_label=current_label,
        mean_label="Historical Mean",
        sort_descending=True,
    )
    event_table_styler = style_event_study_table(
        event_display_table=event_display_table,
        display_unit=resolved_display_unit,
        decimals=table_decimals,
        clip_quantile=table_clip_quantile,
        minimum_clip_value=table_minimum_clip_value,
        clip_mode=table_clip_mode,
    )
    event_heatmap_figure = create_custom_event_heatmap_figure(
        event_display_table=event_display_table,
        asset_label=asset_label,
        display_unit=resolved_display_unit,
        title_label=title_label,
        clip_quantile=table_clip_quantile,
        minimum_clip_value=table_minimum_clip_value,
        clip_mode=table_clip_mode,
        reverse_y_axis=True,
        decimals=table_decimals,
    )

    figure = create_custom_event_seasonality_figure(
        event_path_matrix=event_path_matrix,
        event_metadata=event_metadata,
        asset_label=asset_label,
        title_label=title_label,
        y_axis_title=y_axis_title,
        current_label=current_label,
        mean_label="Historical Mean",
    )

    return {
        "price_series": price_series,
        "event_path_matrix": event_path_matrix,
        "event_metadata": event_metadata,
        "event_display_table": event_display_table,
        "event_table_styler": event_table_styler,
        "event_heatmap_figure": event_heatmap_figure,
        "historical_mean_path": historical_mean_path,
        "current_event_path": current_event_path,
        "summary_stats": summary_stats,
        "figure": figure,
        "config": {
            "asset_label": asset_label,
            "price_col": price_col,
            "change_type": change_type,
            "window_size": window_size,
            "alignment": alignment,
            "require_complete_window": require_complete_window,
            "value_scale": resolved_value_scale,
            "display_unit": resolved_display_unit,
            "title_label": title_label,
            "current_label": current_label,
            "y_axis_title": y_axis_title,
            "table_decimals": table_decimals,
            "table_clip_quantile": table_clip_quantile,
            "table_minimum_clip_value": table_minimum_clip_value,
            "table_clip_mode": table_clip_mode,
        },
    }
