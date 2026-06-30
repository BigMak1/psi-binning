import numpy as np
import pandas as pd

# Метки служебных бинов (значения по умолчанию, не параметры).
NAN_BIN = "missing"
OTHER_BIN = "other"

# Авто-порог min_frequency: clip(AUTO_MIN_FRACTION * n, AUTO_MIN_FLOOR, AUTO_MIN_CEIL). Считается от
# размера reference: на малых выборках мягкий, на больших не отбрасывает надёжные группы.
AUTO_MIN_FRACTION = 0.01
AUTO_MIN_FLOOR = 20
AUTO_MIN_CEIL = 1000


def as_1d(X) -> pd.Series:
    """Приводит вход к одномерной Series.

    Args:
        X: DataFrame с одной колонкой, Series, одномерный массив или список.

    Returns:
        Входные данные в виде Series.

    Raises:
        ValueError: Если в DataFrame больше одной колонки или массив не одномерный.
    """
    if isinstance(X, pd.DataFrame):
        if X.shape[1] != 1:
            raise ValueError(f"ожидается одна фича, в DataFrame {X.shape[1]} колонок")
        return X.iloc[:, 0]
    if isinstance(X, pd.Series):
        return X
    arr = np.asarray(X)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.ravel()
    if arr.ndim != 1:
        raise ValueError(f"ожидается 1D-вектор, получено {arr.ndim}D")
    return pd.Series(arr)


def resolve_min_count(min_frequency: int | float | str | None, n: int) -> int:
    """Преобразует параметр min_frequency в абсолютный порог наблюдений на бин.

    Args:
        min_frequency: Правило порога: ``"auto"`` — clip(AUTO_MIN_FRACTION * n, пол, потолок);
            целое >= 1 — абсолютное значение; float из (0, 1) — доля от ``n``;
            ``None`` или 0 — порог отключён.
        n: Размер выборки, от которого считается порог.

    Returns:
        Абсолютный порог наблюдений на бин.

    Raises:
        ValueError: Если ``min_frequency`` задан неизвестной строкой.
    """
    if not min_frequency:
        return 0
    if isinstance(min_frequency, str):
        if min_frequency == "auto":
            return int(np.clip(round(AUTO_MIN_FRACTION * n), AUTO_MIN_FLOOR, AUTO_MIN_CEIL))
        raise ValueError(f"неизвестное min_frequency: {min_frequency!r}")
    if isinstance(min_frequency, float) and 0.0 < min_frequency < 1.0:
        return int(round(min_frequency * n))
    return int(min_frequency)
