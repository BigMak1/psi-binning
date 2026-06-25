"""Общие хелперы бининга: приведение входа и авто-порог редкого бина."""

import numpy as np
import pandas as pd

# Служебные бины (дефолтные метки, не параметры)
NAN_BIN = "missing"
OTHER_BIN = "other"

# Авто-порог min_bin: clip(AUTO_MIN_FRACTION * n_reference, AUTO_MIN_FLOOR, AUTO_MIN_CEIL).
# Считается ОТ РАЗМЕРА REFERENCE: на мелких выборках мягко, на больших — не выкидывает надёжные группы.
AUTO_MIN_FRACTION = 0.01
AUTO_MIN_FLOOR = 20
AUTO_MIN_CEIL = 1000


def as_1d(X) -> pd.Series:
    """Привести вход к 1D Series: DataFrame одной колонки / Series / 1D-массив / список."""
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


def resolve_min_count(min_bin: int | float | str | None, n: int) -> int:
    """min_bin -> абсолютный порог наблюдений на бин.

    "auto"      — clip(AUTO_MIN_FRACTION * n, AUTO_MIN_FLOOR, AUTO_MIN_CEIL)
    int >= 1    — абсолютное значение
    float (0,1) — доля от n
    None / 0    — порог выключен
    """
    if not min_bin:
        return 0
    if isinstance(min_bin, str):
        if min_bin == "auto":
            return int(np.clip(round(AUTO_MIN_FRACTION * n), AUTO_MIN_FLOOR, AUTO_MIN_CEIL))
        raise ValueError(f"неизвестное min_bin: {min_bin!r}")
    if isinstance(min_bin, float) and 0.0 < min_bin < 1.0:
        return int(round(min_bin * n))
    return int(min_bin)
