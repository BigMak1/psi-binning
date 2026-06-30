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
    binner_min_frequency: int | float | str | None = "auto",
    binner_point_share: float = 0.10,
) -> pd.DataFrame:
    """Вычисляет PSI по периодам для набора фичей.

    Для каждой фичи биннер (``CatBinner`` если фича в ``cat_features``, иначе
    ``NumBinner``) обучается на базе и применяется ко всей выборке, после чего
    считается PSI каждого периода относительно базы. База выбирается один раз и общая
    для всех фичей.

    Args:
        df: Исходный датафрейм.
        features: Имена фичей для расчёта.
        cat_features: Имена категориальных фичей; ``None`` — все фичи числовые.
        psi_date_col: Колонка периода; она же задаёт ось окна базы
            (см. ``define_base_data``).
        psi_alpha: Параметр сглаживания Лапласа.
        psi_base_size: Доля строк в базе; ``None`` — вся выборка.
        psi_base_shift_size: Сдвиг окна базы по времени (доля).
        psi_base_mask_col: Имя готовой булевой колонки базы (приоритет над size/shift).
        binner_n_bins: Число квантильных интервалов (для числовых фичей).
        binner_min_frequency: Порог редкого бина (см. ``resolve_min_count``).
        binner_point_share: Порог доли для выделения точечного бина (для числовых
            фичей).

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

    psi_by_features = {}
    for feature in features:
        if feature in cat_features:
            binner = CatBinner(min_frequency=binner_min_frequency)
        else:
            binner = NumBinner(
                n_bins=binner_n_bins,
                min_frequency=binner_min_frequency,
                point_share=binner_point_share,
            )
        binner.fit(df.loc[base_mask, feature])  # учим границы только на базе
        binned_feature = binner.transform(df[feature])  # применяем ко всей выборке
        psi_by_features[feature] = calc_psi_by_period(
            df, binned_feature, psi_date_col, base_mask, alpha=psi_alpha
        )
    psi_by_features = pd.DataFrame(psi_by_features).reset_index(names=[psi_date_col])
    base_dates = df.loc[base_mask, psi_date_col].unique()
    psi_by_features["is_psi_base"] = psi_by_features[psi_date_col].isin(base_dates)
    return psi_by_features


def calc_bin_counts_by_period(
    binned_feature: pd.Series, periods: pd.Series
) -> pd.DataFrame:
    """Считает число наблюдений в каждом бине по периодам.

    Порядок строк берётся из ``cat.categories``: смешанные бины (Interval/число/строка)
    несравнимы, обычная сортировка по ним падает. В результат включаются ВСЕ
    категории, в том числе глобально пустые (например, ``missing`` без пропусков) —
    с нулями: это держит единый универсум бинов, на котором PSI считается матрицей
    разом.

    Индексы обоих аргументов сбрасываются, поэтому выравнивание идёт строго по позиции
    (а не по индексу, как сделал бы ``crosstab`` сам) — функцию можно звать с любыми
    Series.

    Args:
        binned_feature: Категориальная Series бинов (результат ``transform``).
        periods: Series периодов той же длины (выравнивается по позиции).

    Returns:
        DataFrame счётчиков: индекс — бин (в порядке категорий), колонки — период.
    """
    binned_feature = binned_feature.reset_index(drop=True)
    periods = periods.reset_index(drop=True)
    counts = pd.crosstab(binned_feature, periods)
    return counts.reindex(binned_feature.cat.categories, fill_value=0)


def calc_psi_by_period(
    df: pd.DataFrame,
    binned_feature: pd.Series,
    psi_date_col: str,
    base_mask: np.ndarray,
    *,
    alpha: float = 0.5,
) -> pd.Series:
    """Вычисляет PSI каждого периода относительно базового распределения.

    Счётчики бинов по периодам берутся разом матрицей ``calc_bin_counts_by_period``
    (бин × период), база — общим распределением по ``base_mask``; PSI всех периодов
    получается одной векторной операцией без цикла по периодам.

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
    actual = calc_bin_counts_by_period(binned_feature, df[psi_date_col])  # бин × период
    expected = (
        binned_feature[base_mask].value_counts().reindex(actual.index, fill_value=0)
    )
    psi = _psi_laplace(expected.to_numpy(float), actual.to_numpy(float), alpha)
    return pd.Series(psi, index=actual.columns, name="psi").sort_index()


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
        mask_col: Имя готовой булевой колонки базы (приоритет над остальными
            параметрами).

    Returns:
        Булева маска строк базы той же длины, что и ``df``.

    Examples:
        define_base_data(df)                                 -> вся выборка
        define_base_data(df, base_size=0.2)                  -> первые 20% по дате
        define_base_data(df, base_size=0.2, shift_size=0.2)  -> вторые 20% (20–40%)
        define_base_data(df, mask_col="is_base")             -> готовая разметка
    """
    n = len(df)
    if mask_col is not None:
        return df[mask_col].to_numpy(dtype=bool)
    if base_size is None:
        return np.ones(n, dtype=bool)
    order = np.argsort(df[date_col].to_numpy(), kind="stable")
    lo = min(int(shift_size * n), n)
    hi = min(lo + int(base_size * n), n)
    mask = np.zeros(n, dtype=bool)
    mask[order[lo:hi]] = True
    return mask


def _psi_laplace(
    expected: np.ndarray, actual: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """Вычисляет PSI между распределениями бинов со сглаживанием Лапласа.

    Доля бина = (count + alpha) / (N + alpha * B). Сглаживание убирает нули и деление на
    ноль; при больших N влияние alpha ничтожно. alpha = 0.5 соответствует приору
    Джеффриса. Считается векторно по столбцам: ``expected`` — счётчики базы (B,),
    ``actual`` — счётчики бинов по периодам (B, P); PSI получается для всех периодов
    сразу.

    Args:
        expected: Базовые счётчики бинов, форма (B,).
        actual: Счётчики бинов по периодам, форма (B, P).
        alpha: Параметр сглаживания Лапласа.

    Returns:
        Значения PSI по периодам, форма (P,).
    """
    num_bins = expected.shape[0]
    expected = (expected[:, None] + alpha) / (expected.sum() + alpha * num_bins)
    actual = (actual + alpha) / (actual.sum(axis=0) + alpha * num_bins)
    return ((actual - expected) * np.log(actual / expected)).sum(axis=0)


def _psi_epsilon(
    expected: np.ndarray, actual: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    """Вычисляет PSI между распределениями бинов с epsilon-клиппингом.

    Доли считаются напрямую и поднимаются до ``eps`` (clip), чтобы убрать log(0) и
    деление на ноль. Классическая альтернатива сглаживанию Лапласа (``_psi_laplace``);
    те же формы входов/выхода — векторно по столбцам.

    Args:
        expected: Базовые счётчики бинов, форма (B,).
        actual: Счётчики бинов по периодам, форма (B, P).
        eps: Нижняя граница доли.

    Returns:
        Значения PSI по периодам, форма (P,).
    """
    expected = np.clip(expected[:, None] / expected.sum(), eps, None)
    actual = np.clip(actual / actual.sum(axis=0), eps, None)
    return ((actual - expected) * np.log(actual / expected)).sum(axis=0)
