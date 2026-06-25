"""Категориальный биннинг: класс CategoricalBinner (fit/transform).

fit учит points_ — частые категории (>= min_bin наблюдений, по убыванию частоты): каждая получает
свой бин. Редкие и новые категории при transform уходят в OTHER_BIN, NaN -> NAN_BIN (чтобы не
раздували PSI).

Состояние после fit: points_ (частые категории). После transform: bins_ — упорядоченные идентичности
бинов (points_ + other + missing). Класс ничего не знает про PSI — годится и для WoE.
"""

from typing import Self

import pandas as pd

from .common import NAN_BIN, OTHER_BIN, as_1d, resolve_min_count


class CatBinner:
    """Бьёт категориальную фичу на бины. Учит частые категории на переданных данных (база/train)."""

    def __init__(self, *, min_bin: int | float | str | None = "auto"):
        self.min_bin = min_bin

    def fit(self, X) -> Self:
        """Выучить points_ — частые категории (>= min_bin наблюдений), по убыванию частоты."""
        X = as_1d(X).astype("object").dropna()
        counts = X.value_counts()
        min_count = resolve_min_count(self.min_bin, len(X))
        self.points_ = [category for category, count in counts.items() if count >= min_count]
        return self

    def transform(self, X) -> pd.Series:
        """Свести значения к выученным точкам. Редкие/новые -> OTHER_BIN, NaN -> NAN_BIN.

        Возвращает упорядоченную категориальную Series (точки по частоте, затем other, missing).
        """
        self._check_fitted()
        X = as_1d(X).reset_index(drop=True).astype("object")

        out = X.where(X.isin(set(self.points_)), OTHER_BIN)   # редкие/новые -> other
        out[X.isna()] = NAN_BIN                                # пропуски -> missing

        out = pd.Series(pd.Categorical(out, categories=[*self.points_, OTHER_BIN, NAN_BIN], ordered=True))
        self.bins_ = out.cat.categories
        return out

    def fit_transform(self, X) -> pd.Series:
        return self.fit(X).transform(X)

    def _check_fitted(self) -> None:
        if not hasattr(self, "points_"):
            raise RuntimeError("CategoricalBinner не обучен: вызовите fit() перед transform().")
