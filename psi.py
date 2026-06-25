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
    """Вычисляет PSI по периодам для набора фичей.

    Для каждой фичи биннер (``CatBinner`` если фича в ``cat_features``, иначе ``NumBinner``)
    обучается на базе и применяется ко всей выборке, после чего считается PSI каждого периода
    относительно базы. База выбирается один раз и общая для всех фичей.

    Args:
        df: Исходный датафрейм.
        features: Имена фичей для расчёта.
        cat_features: Имена категориальных фичей; ``None`` — все фичи числовые.
        psi_date_col: Колонка периода; она же задаёт ось окна базы (см. ``define_base_data``).
        psi_alpha: Параметр сглаживания Лапласа.
        psi_base_size: Доля строк в базе; ``None`` — вся выборка.
        psi_base_shift_size: Сдвиг окна базы по времени (доля).
        psi_base_mask_col: Имя готовой булевой колонки базы (приоритет над size/shift).
        binner_n_bins: Число квантильных интервалов (для числовых фичей).
        binner_min_bin: Порог редкого бина (см. ``resolve_min_count``).
        binner_point_share: Порог доли для выделения точечного бина (для числовых фичей).

    Returns:
        DataFrame со значениями PSI: индекс — период, колонки — фичи.
    """
    base_mask = define_base_data(
        df,
        date_col=psi_date_col,
        base_size=psi_base_size,
        shift_size=psi_base_shift_size,
        mask_col=psi_base_mask_col,
    )
    cat_features = set(cat_features or [])  # пусто -> все фичи числовые

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
        binner.fit(df.loc[base_mask, feature])  # учим границы только на базе
        binned_feature = binner.transform(df[feature])  # применяем ко всей выборке
        psi_by_feature[feature] = calc_psi_by_period(
            df, binned_feature, psi_date_col, base_mask, alpha=psi_alpha
        )

    return pd.DataFrame(psi_by_feature)


def calc_psi_by_period(
    df: pd.DataFrame,
    binned_feature: pd.Series,
    psi_date_col: str,
    base_mask: np.ndarray,
    *,
    alpha: float = 0.5,
) -> pd.Series:
    """Вычисляет PSI каждого периода относительно базового распределения.

    Args:
        df: Исходный датафрейм; из него берётся колонка периода.
        binned_feature: Категориальная Series бинов (результат transform), выровненная
            по позиции с ``df``.
        psi_date_col: Имя колонки периода в ``df`` (например, ``"sample_month"``).
        base_mask: Булева маска базовых строк той же длины, что и ``df``.
        alpha: Параметр сглаживания Лапласа.

    Returns:
        Series со значениями PSI, индексированная и отсортированная по периоду.
    """
    expected = binned_feature[base_mask].value_counts().to_dict()

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
    """Строит булеву (позиционную) маску строк базы (reference).

    Args:
        df: Исходный датафрейм.
        date_col: Колонка даты, по которой сортируется окно базы.
        base_size: Доля строк в базе; ``None`` — вся выборка.
        shift_size: Сдвиг окна базы по времени (доля): окно строк [shift, shift + base];
            позволяет пропустить проблемные ранние периоды.
        mask_col: Имя готовой булевой колонки базы (приоритет над остальными параметрами).

    Returns:
        Булева маска строк базы той же длины, что и ``df``.

    Examples:
        define_base_data(df)                                 -> вся выборка
        define_base_data(df, base_size=0.2)                  -> первые 20% по дате
        define_base_data(df, base_size=0.2, shift_size=0.2)  -> вторые 20% (с 20% по 40%)
        define_base_data(df, mask_col="is_base")             -> готовая разметка
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
    """Вычисляет PSI между распределениями бинов со сглаживанием Лапласа.

    Доля бина = (count + alpha) / (N + alpha * B). Сглаживание убирает нули и деление на ноль;
    при больших N влияние alpha ничтожно. alpha = 0.5 соответствует приору Джеффриса.

    Args:
        expected: Базовое распределение, словарь bin -> count.
        actual: Сравниваемое распределение, словарь bin -> count.
        alpha: Параметр сглаживания Лапласа.

    Returns:
        Значение PSI.
    """
    # Ключи-бины разнотипны (Interval / число / строка), поэтому сортируем по строковому виду.
    bins = sorted(set(expected) | set(actual), key=str)
    num_bins = len(bins)
    expected = np.array([expected.get(k, 0.0) for k in bins], dtype=float)
    actual = np.array([actual.get(k, 0.0) for k in bins], dtype=float)
    expected = (expected + alpha) / (expected.sum() + alpha * num_bins)
    actual = (actual + alpha) / (actual.sum() + alpha * num_bins)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def _psi_epsilon(expected: dict, actual: dict, eps: float = 1e-6) -> float:
    """Вычисляет PSI между распределениями бинов с epsilon-клиппингом.

    Доли считаются напрямую и поднимаются до ``eps`` (clip), чтобы убрать log(0) и деление
    на ноль. Классическая альтернатива сглаживанию Лапласа (``_psi_laplace``).

    Args:
        expected: Базовое распределение, словарь bin -> count.
        actual: Сравниваемое распределение, словарь bin -> count.
        eps: Нижняя граница доли.

    Returns:
        Значение PSI.
    """
    bins = sorted(set(expected) | set(actual), key=str)
    expected = np.array([expected.get(k, 0.0) for k in bins], dtype=float)
    actual = np.array([actual.get(k, 0.0) for k in bins], dtype=float)
    expected = np.clip(expected / expected.sum(), eps, None)
    actual = np.clip(actual / actual.sum(), eps, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))
