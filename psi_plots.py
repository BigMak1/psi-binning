from collections.abc import Sequence

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from binning import CatBinner, NumBinner
from psi import calc_bin_counts_by_period, calc_psi_by_period, define_base_data

# Пороги интерпретации PSI и их оформление на графике (y, подпись, цвет, штрих).
PSI_WARN = 0.10
PSI_ALERT = 0.25
PSI_THRESHOLDS = (
    (PSI_WARN, "0.10 warn", "orange", "dot"),
    (PSI_ALERT, "0.25 alert", "red", "dash"),
)


def _base_caption(
    psi_base_size: float | None,
    psi_base_shift_size: float,
    psi_base_mask_col: str | None,
) -> str:
    """Формирует подпись об источнике базы для заголовка графика."""
    if psi_base_mask_col is not None:
        return f"база: {psi_base_mask_col}"
    if psi_base_size is None:
        return "база: вся выборка"
    shift = f" (сдвиг {psi_base_shift_size:.0%})" if psi_base_shift_size else ""
    return f"база: {psi_base_size:.0%}{shift}"


def _fit_binner(
    df,
    feature,
    *,
    is_category,
    binner_n_bins,
    binner_min_frequency,
    binner_point_share,
    psi_date_col,
    psi_base_size,
    psi_base_shift_size,
    psi_base_mask_col,
):
    """Выбирает базу, обучает биннер на ней и применяет ко всей выборке.

    Returns:
        Кортеж (binned_feature, base_mask).
    """
    base_mask = define_base_data(
        df,
        date_col=psi_date_col,
        base_size=psi_base_size,
        shift_size=psi_base_shift_size,
        mask_col=psi_base_mask_col,
    )
    binner = (
        CatBinner(min_frequency=binner_min_frequency)
        if is_category
        else NumBinner(
            n_bins=binner_n_bins,
            min_frequency=binner_min_frequency,
            point_share=binner_point_share,
        )
    )
    binner.fit(df.loc[base_mask, feature])
    return binner.transform(df[feature]), base_mask


def _add_period_lines(
    fig: go.Figure,
    table: pd.DataFrame | pd.Series,
    *,
    normalize: bool = False,
    color: str | None = None,
    row=None,
    col=None,
) -> None:
    """Добавляет в ``fig`` по линии (lines+markers) на каждый ряд таблицы.

    ``Series`` трактуется как одна линия; ``DataFrame`` — строки = линии, колонки =
    периоды (ось X). ``normalize`` нормирует столбцы в доли; полностью нулевые/пустые
    ряды не рисуются (чтобы не плодить линии-нули, например ``missing`` без пропусков).
    """
    if isinstance(table, pd.Series):
        table = table.to_frame().T
    table = table.loc[(table.fillna(0) != 0).any(axis=1)]
    if normalize:
        table = table / table.sum(axis=0)
    periods = list(table.columns)
    line = {"color": color} if color else None
    for name in table.index:
        fig.add_scatter(
            x=periods,
            y=table.loc[name].to_numpy(),
            mode="lines+markers",
            name=str(name),
            line=line,
            row=row,
            col=col,
        )


def _add_thresholds(
    fig: go.Figure,
    thresholds: Sequence[tuple],
    *,
    annotate: bool = True,
    row=None,
    col=None,
) -> None:
    """Рисует горизонтальные линии-пороги (y, подпись, цвет, тип штриха)."""
    for y, label, color, dash in thresholds:
        ann = (
            {"annotation_text": label, "annotation_position": "top left"}
            if annotate
            else {}
        )
        fig.add_hline(y=y, line_dash=dash, line_color=color, row=row, col=col, **ann)


def draw_lines_by_period(
    table: pd.DataFrame | pd.Series,
    *,
    normalize: bool = False,
    thresholds: Sequence[tuple] = (),
    color: str | None = None,
    title: str = "",
    xaxis_title: str = "Период",
    yaxis_title: str = "",
    legend_title: str = "",
    hovermode: str = "x unified",
    height: int | None = None,
) -> go.Figure:
    """Рисует линии по периодам — по линии на каждый ряд таблицы.

    Универсальный примитив: ``DataFrame`` (строки = бины/ряды, колонки = периоды) даёт
    линию на строку, ``Series`` (индекс = периоды) — одну линию (например, PSI). Бины и
    PSI рисуются одним и тем же кодом; различие — только в данных и оформлении.

    Args:
        table: Таблица бин × период (или Series ряда по периодам).
        normalize: Нормировать столбцы в доли (счётчики -> доли периода).
        thresholds: Пороги-hline: кортежи (y, подпись, цвет, штрих).
        color: Единый цвет линий (для одиночного ряда вроде PSI); ``None`` — палитра
            по умолчанию.
        title: Заголовок; xaxis_title/yaxis_title/legend_title/hovermode/height —
            оформление.

    Returns:
        Figure plotly с линиями по периодам.
    """
    fig = go.Figure()
    _add_period_lines(fig, table, normalize=normalize, color=color)
    _add_thresholds(fig, thresholds)
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        legend_title=legend_title,
        hovermode=hovermode,
        height=height,
    )
    return fig


def plot_bin_distribution(
    df,
    feature,
    *,
    is_category=False,
    binner_n_bins=10,
    binner_min_frequency="auto",
    binner_point_share=0.10,
    psi_date_col="sample_month",
    psi_base_size=None,
    psi_base_shift_size=0.0,
    psi_base_mask_col=None,
) -> go.Figure:
    """Строит линии долей бинов по периодам — по линии на каждый бин.

    Параметры биннинга и выбора базы совпадают с ``psi.calc_psi_by_features``.

    Returns:
        Figure plotly с линиями долей бинов.
    """
    binned_feature, _ = _fit_binner(
        df,
        feature,
        is_category=is_category,
        binner_n_bins=binner_n_bins,
        binner_min_frequency=binner_min_frequency,
        binner_point_share=binner_point_share,
        psi_date_col=psi_date_col,
        psi_base_size=psi_base_size,
        psi_base_shift_size=psi_base_shift_size,
        psi_base_mask_col=psi_base_mask_col,
    )
    counts = calc_bin_counts_by_period(binned_feature, df[psi_date_col])
    return draw_lines_by_period(
        counts,
        normalize=True,
        title=f"Распределение бинов — {feature}",
        yaxis_title="Доля",
        legend_title="Бин",
    )


def plot_psi(
    df,
    feature,
    *,
    is_category=False,
    binner_n_bins=10,
    binner_min_frequency="auto",
    binner_point_share=0.10,
    psi_date_col="sample_month",
    psi_base_size=None,
    psi_base_shift_size=0.0,
    psi_base_mask_col=None,
    psi_alpha=0.5,
) -> go.Figure:
    """Строит линию PSI по периодам с порогами 0.10 / 0.25.

    Параметры биннинга и выбора базы совпадают с ``psi.calc_psi_by_features``.

    Returns:
        Figure plotly с линией PSI и линиями порогов.
    """
    binned_feature, base_mask = _fit_binner(
        df,
        feature,
        is_category=is_category,
        binner_n_bins=binner_n_bins,
        binner_min_frequency=binner_min_frequency,
        binner_point_share=binner_point_share,
        psi_date_col=psi_date_col,
        psi_base_size=psi_base_size,
        psi_base_shift_size=psi_base_shift_size,
        psi_base_mask_col=psi_base_mask_col,
    )
    psi = calc_psi_by_period(
        df, binned_feature, psi_date_col, base_mask, alpha=psi_alpha
    )
    cap = _base_caption(psi_base_size, psi_base_shift_size, psi_base_mask_col)
    return draw_lines_by_period(
        psi.rename("PSI"),
        thresholds=PSI_THRESHOLDS,
        color="#1f2937",
        title=f"PSI по периодам — {feature} ({cap})",
        yaxis_title="PSI",
    )


def plot_feature(
    df,
    feature,
    *,
    is_category=False,
    binner_n_bins=10,
    binner_min_frequency="auto",
    binner_point_share=0.10,
    psi_date_col="sample_month",
    psi_base_size=None,
    psi_base_shift_size=0.0,
    psi_base_mask_col=None,
    psi_alpha=0.5,
    height=640,
) -> go.Figure:
    """Строит комбинированный график: сверху доли бинов, снизу PSI по периодам.

    Параметры биннинга и выбора базы совпадают с ``psi.calc_psi_by_features``.

    Returns:
        Figure plotly из двух подграфиков (распределение и PSI).
    """
    binned_feature, base_mask = _fit_binner(
        df,
        feature,
        is_category=is_category,
        binner_n_bins=binner_n_bins,
        binner_min_frequency=binner_min_frequency,
        binner_point_share=binner_point_share,
        psi_date_col=psi_date_col,
        psi_base_size=psi_base_size,
        psi_base_shift_size=psi_base_shift_size,
        psi_base_mask_col=psi_base_mask_col,
    )
    counts = calc_bin_counts_by_period(binned_feature, df[psi_date_col])
    psi = calc_psi_by_period(
        df, binned_feature, psi_date_col, base_mask, alpha=psi_alpha
    )
    cap = _base_caption(psi_base_size, psi_base_shift_size, psi_base_mask_col)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.62, 0.38],
        vertical_spacing=0.09,
        subplot_titles=(f"Доли бинов — {feature}", f"PSI ({cap})"),
    )
    _add_period_lines(fig, counts, normalize=True, row=1, col=1)
    _add_period_lines(fig, psi.rename("PSI"), color="#1f2937", row=2, col=1)
    _add_thresholds(fig, PSI_THRESHOLDS, annotate=False, row=2, col=1)

    fig.update_layout(
        height=height, title=f"PSI-обзор: {feature}", legend={"title": "Бин / PSI"}
    )
    fig.update_yaxes(title_text="Доля", row=1, col=1)
    fig.update_yaxes(title_text="PSI", row=2, col=1)
    fig.update_xaxes(title_text="Период", row=2, col=1)
    return fig
