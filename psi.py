"""Расчёт PSI по периодам поверх биннеров (NumBinner / CatBinner).

PSI = sum_i (actual_i - expected_i) * ln(actual_i / expected_i).
Пороги: < 0.10 стабильно, 0.10..0.25 — сдвиг, > 0.25 — сильный сдвиг.

Иерархия (сверху вниз):
    calc_psi_by_features  — PSI по периодам для набора фичей (оркестратор: база -> биннинг -> PSI)
    calc_psi_by_period    — PSI каждого периода относительно базы для одной фичи
    define_base_data      — выбор базы (reference): булева маска строк
    _psi_laplace          — PSI из counts, сглаживание Лапласа (используется по умолчанию)
    _psi_epsilon          — PSI из counts, epsilon-клиппинг (альтернатива)

Параметры calc_psi_by_features сгруппированы префиксами:
    binner_*    — как бить на бины (идут в NumBinner / CatBinner)
    psi_base_*  — как выбрать базу (идут в define_base_data)
"""

import numpy as np
import pandas as pd

from binning import CatBinner, NumBinner


def calc_psi_by_features(
    df: pd.DataFrame,
    features: list[str],
    *,
    cat_features: list[str] | None = None,
    psi_date_col: str = "sample_month",
    psi_alpha: float = 0.5,
    psi_base_size: float | None = None,
    psi_base_shift_size: float = 0.0,
    psi_base_mask_col: str | None = None,
    binner_n_bins: int = 10,
    binner_min_bin: int | float | str | None = "auto",
    binner_point_share: float = 0.10,
) -> pd.DataFrame:
    """PSI по периодам для набора фичей. -> DataFrame: index=period, columns=features.

    Для КАЖДОЙ фичи: биннер (CatBinner если фича в cat_features, иначе NumBinner)
    обучается на базе и применяется ко всей выборке; затем считаем PSI каждого периода
    относительно базы. База выбирается один раз (define_base_data) и общая для всех фичей.
    """
    base_mask = define_base_data(
        df,
        date_col=psi_date_col,
        base_size=psi_base_size,
        shift_size=psi_base_shift_size,
        mask_col=psi_base_mask_col,
    )
    cat_features = set(cat_features or [])   # нет категориальных -> пустое множество, все фичи числовые

    psi_by_feature = {}
    for feature in features:
        if feature in cat_features:
            binner = CatBinner(min_bin=binner_min_bin)
        else:
            binner = NumBinner(
                n_bins=binner_n_bins,
                min_bin=binner_min_bin,
                point_share=binner_point_share,
            )
        binner.fit(df.loc[base_mask, feature])             # учим границы только на базе
        binned_feature = binner.transform(df[feature])          # применяем ко всей выборке
        psi_by_feature[feature] = calc_psi_by_period(df, binned_feature, psi_date_col, base_mask, alpha=psi_alpha)

    return pd.DataFrame(psi_by_feature)


def calc_psi_by_period(
    df: pd.DataFrame,
    binned_feature: pd.Series,
    psi_date_col: str,
    base_mask: np.ndarray,
    *,
    alpha: float = 0.5,
) -> pd.Series:
    """PSI каждого периода относительно базы для одной фичи. -> Series: index=period, value=PSI.

    df             — исходный датафрейм (из него берём колонку периода psi_date_col).
    binned_feature — категориальная Series бинов (выход transform), выровнена ПО ПОЗИЦИИ с df.
    psi_date_col   — имя колонки периода в df (напр. "sample_month").
    base_mask      — булева маска базы той же длины: expected = распределение бинов на базе.
    """
    expected = binned_feature[base_mask].value_counts().to_dict()   # распределение бинов на базе

    psi = {}
    for period, period_bins in binned_feature.groupby(df[psi_date_col].to_numpy()):
        actual = period_bins.value_counts().to_dict()
        psi[period] = _psi_laplace(expected, actual, alpha)

    return pd.Series(psi, name="psi").sort_index()


def define_base_data(
    df: pd.DataFrame,
    *,
    date_col: str = "sample_date_orig",
    base_size: float | None = None,
    shift_size: float = 0.0,
    mask_col: str | None = None,
) -> np.ndarray:
    """Булева (позиционная) маска строк базы (reference) для PSI.

    mask_col   — если задан, берём готовую bool-колонку из df (приоритет над остальным).
    base_size  — доля строк в базе по сортировке date_col; None -> вся выборка.
    shift_size — сдвиг окна базы по времени (доля): окно строк [shift, shift + base];
                 позволяет пропустить проблемные ранние периоды.

    Примеры:
        define_base_data(df)                                -> вся выборка
        define_base_data(df, base_size=0.2)                 -> первые 20% по дате
        define_base_data(df, base_size=0.2, shift_size=0.2) -> вторые 20% (с 20% по 40%)
        define_base_data(df, mask_col="is_base")            -> готовая разметка пользователя
    """
    n = len(df)
    if mask_col is not None:
        return df[mask_col].to_numpy(dtype=bool)
    if base_size is None:
        return np.ones(n, dtype=bool)
    order = np.argsort(df[date_col].to_numpy(), kind="stable")
    lo = min(int(round(shift_size * n)), n)
    hi = min(lo + int(round(base_size * n)), n)
    mask = np.zeros(n, dtype=bool)
    mask[order[lo:hi]] = True
    return mask


def _psi_laplace(expected: dict, actual: dict, alpha: float = 0.5) -> float:
    """PSI между распределениями бинов; expected/actual — словари bin -> COUNT.

    Сглаживание Лапласа: доля = (count + alpha) / (N + alpha * B). Убирает нули/деление-на-ноль
    статистически корректно; при больших N влияние alpha ничтожно. alpha=0.5 — Джеффрис.
    """
    bins = sorted(set(expected) | set(actual), key=str)   # ключи-бины разнотипны (Interval/число/str)
    num_bins = len(bins)
    expected = np.array([expected.get(k, 0.0) for k in bins], dtype=float)
    actual = np.array([actual.get(k, 0.0) for k in bins], dtype=float)
    expected = (expected + alpha) / (expected.sum() + alpha * num_bins)
    actual = (actual + alpha) / (actual.sum() + alpha * num_bins)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def _psi_epsilon(expected: dict, actual: dict, eps: float = 1e-6) -> float:
    """PSI между распределениями бинов; expected/actual — словари bin -> COUNT.

    Классический epsilon-вариант: доли считаем напрямую и поднимаем до eps (clip),
    чтобы убрать log(0)/деление на ноль. Альтернатива Лапласу (_psi_laplace).
    """
    bins = sorted(set(expected) | set(actual), key=str)
    expected = np.array([expected.get(k, 0.0) for k in bins], dtype=float)
    actual = np.array([actual.get(k, 0.0) for k in bins], dtype=float)
    expected = np.clip(expected / expected.sum(), eps, None)
    actual = np.clip(actual / actual.sum(), eps, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))
