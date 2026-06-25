import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from binning import CatBinner, NumBinner
from psi import calc_psi_by_period, define_base_data

# Пороги интерпретации PSI.
PSI_WARN = 0.10
PSI_ALERT = 0.25


def _base_caption(
    psi_base_size: float | None, psi_base_shift_size: float, psi_base_mask_col: str | None
) -> str:
    """Формирует подпись об источнике базы для заголовка графика."""
    if psi_base_mask_col is not None:
        return f"база: {psi_base_mask_col}"
    if psi_base_size is None:
        return "база: вся выборка"
    shift = f" (сдвиг {psi_base_shift_size:.0%})" if psi_base_shift_size else ""
    return f"база: {psi_base_size:.0%}{shift}"


def _fit_binner(
    df, feature, *, is_category,
    binner_n_bins, binner_min_bin, binner_point_share,
    psi_date_col, psi_base_size, psi_base_shift_size, psi_base_mask_col,
):
    """Выбирает базу, обучает биннер на ней и применяет ко всей выборке.

    Returns:
        Кортеж (binned_feature, base_mask).
    """
    base_mask = define_base_data(df, date_col=psi_date_col, base_size=psi_base_size,
                                 shift_size=psi_base_shift_size, mask_col=psi_base_mask_col)
    binner = (CatBinner(min_bin=binner_min_bin) if is_category
              else NumBinner(n_bins=binner_n_bins, min_bin=binner_min_bin,
                             point_share=binner_point_share))
    binner.fit(df.loc[base_mask, feature])
    return binner.transform(df[feature]), base_mask


def _bin_shares(binned_feature: pd.Series, periods: pd.Series) -> pd.DataFrame:
    """Считает доли бинов по периодам (индекс — бин, колонки — период).

    Группировка идёт по самой категориальной Series (через crosstab по кодам): смешанные
    бины (Interval/число/строка) несравнимы, обычная сортировка по ним падает.
    """
    share = pd.crosstab(binned_feature, periods, normalize="columns")
    order = [b for b in binned_feature.cat.categories if b in share.index]
    return share.reindex(order)


def plot_bin_distribution(
    df, feature, *, is_category=False,
    binner_n_bins=10, binner_min_bin="auto", binner_point_share=0.10,
    psi_date_col="sample_month", psi_base_size=None,
    psi_base_shift_size=0.0, psi_base_mask_col=None,
) -> go.Figure:
    """Строит линии долей бинов по периодам — по линии на каждый бин.

    Параметры биннинга и выбора базы совпадают с ``psi.calc_psi_by_features``.

    Returns:
        Figure plotly с линиями долей бинов.
    """
    binned_feature, _ = _fit_binner(
        df, feature, is_category=is_category,
        binner_n_bins=binner_n_bins, binner_min_bin=binner_min_bin,
        binner_point_share=binner_point_share, psi_date_col=psi_date_col,
        psi_base_size=psi_base_size, psi_base_shift_size=psi_base_shift_size,
        psi_base_mask_col=psi_base_mask_col,
    )
    share = _bin_shares(binned_feature, pd.Series(df[psi_date_col].to_numpy()))
    periods = list(share.columns)
    fig = go.Figure()
    for b in share.index:
        fig.add_scatter(x=periods, y=share.loc[b].to_numpy(), mode="lines+markers", name=str(b))
    fig.update_layout(
        title=f"Распределение бинов — {feature}",
        xaxis_title="Период", yaxis_title="Доля", legend_title="Бин", hovermode="x unified",
    )
    return fig


def plot_psi(
    df, feature, *, is_category=False,
    binner_n_bins=10, binner_min_bin="auto", binner_point_share=0.10,
    psi_date_col="sample_month", psi_base_size=None,
    psi_base_shift_size=0.0, psi_base_mask_col=None,
    psi_alpha=0.5,
) -> go.Figure:
    """Строит линию PSI по периодам с порогами 0.10 / 0.25.

    Параметры биннинга и выбора базы совпадают с ``psi.calc_psi_by_features``.

    Returns:
        Figure plotly с линией PSI и линиями порогов.
    """
    binned_feature, base_mask = _fit_binner(
        df, feature, is_category=is_category,
        binner_n_bins=binner_n_bins, binner_min_bin=binner_min_bin,
        binner_point_share=binner_point_share, psi_date_col=psi_date_col,
        psi_base_size=psi_base_size, psi_base_shift_size=psi_base_shift_size,
        psi_base_mask_col=psi_base_mask_col,
    )
    psi = calc_psi_by_period(df, binned_feature, psi_date_col, base_mask, alpha=psi_alpha)
    fig = go.Figure()
    fig.add_scatter(x=list(psi.index), y=psi.to_numpy(),
                    mode="lines+markers", name="PSI", line=dict(color="#1f2937"))
    fig.add_hline(y=PSI_WARN, line_dash="dot", line_color="orange",
                  annotation_text="0.10 warn", annotation_position="top left")
    fig.add_hline(y=PSI_ALERT, line_dash="dash", line_color="red",
                  annotation_text="0.25 alert", annotation_position="top left")
    cap = _base_caption(psi_base_size, psi_base_shift_size, psi_base_mask_col)
    fig.update_layout(title=f"PSI по периодам — {feature} ({cap})",
                      xaxis_title="Период", yaxis_title="PSI")
    return fig


def plot_feature(
    df, feature, *, is_category=False,
    binner_n_bins=10, binner_min_bin="auto", binner_point_share=0.10,
    psi_date_col="sample_month", psi_base_size=None,
    psi_base_shift_size=0.0, psi_base_mask_col=None,
    psi_alpha=0.5, height=640,
) -> go.Figure:
    """Строит комбинированный график: сверху доли бинов, снизу PSI по периодам.

    Параметры биннинга и выбора базы совпадают с ``psi.calc_psi_by_features``.

    Returns:
        Figure plotly из двух подграфиков (распределение и PSI).
    """
    binned_feature, base_mask = _fit_binner(
        df, feature, is_category=is_category,
        binner_n_bins=binner_n_bins, binner_min_bin=binner_min_bin,
        binner_point_share=binner_point_share, psi_date_col=psi_date_col,
        psi_base_size=psi_base_size, psi_base_shift_size=psi_base_shift_size,
        psi_base_mask_col=psi_base_mask_col,
    )
    share = _bin_shares(binned_feature, pd.Series(df[psi_date_col].to_numpy()))
    psi = calc_psi_by_period(df, binned_feature, psi_date_col, base_mask, alpha=psi_alpha)
    psi = psi.reindex(share.columns)  # выровнять по той же оси периодов, что и распределение
    periods = list(share.columns)
    cap = _base_caption(psi_base_size, psi_base_shift_size, psi_base_mask_col)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.09,
        subplot_titles=(f"Доли бинов — {feature}", f"PSI ({cap})"),
    )
    for b in share.index:
        fig.add_scatter(x=periods, y=share.loc[b].to_numpy(), mode="lines+markers",
                        name=str(b), row=1, col=1)
    fig.add_scatter(x=periods, y=psi.to_numpy(), mode="lines+markers",
                    name="PSI", line=dict(color="#1f2937"), row=2, col=1)
    fig.add_hline(y=PSI_WARN, line_dash="dot", line_color="orange", row=2, col=1)
    fig.add_hline(y=PSI_ALERT, line_dash="dash", line_color="red", row=2, col=1)

    fig.update_layout(height=height, title=f"PSI-обзор: {feature}", legend=dict(title="Бин / PSI"))
    fig.update_yaxes(title_text="Доля", row=1, col=1)
    fig.update_yaxes(title_text="PSI", row=2, col=1)
    fig.update_xaxes(title_text="Период", row=2, col=1)
    return fig
