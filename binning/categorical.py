from typing import Self

import pandas as pd

from .common import NAN_BIN, OTHER_BIN, as_1d, resolve_min_count


class CatBinner:
    """Разбивает категориальную фичу на бины.

    Учит частые категории на переданных данных (база/train): каждая получает свой бин.
    При transform редкие и новые категории уходят в ``OTHER_BIN``, пропуски — в ``NAN_BIN``.
    Класс не знает про PSI и пригоден, например, для WoE.
    """

    def __init__(self, *, min_frequency: int | float | str | None = "auto"):
        self.min_frequency = min_frequency

    def fit(self, X) -> Self:
        """Учит частые категории points_ (>= min_frequency наблюдений, по убыванию частоты).

        Args:
            X: Категориальная фича (база/train).

        Returns:
            Сам обученный биннер.
        """
        X = as_1d(X).astype("object").dropna()
        counts = X.value_counts()
        min_count = resolve_min_count(self.min_frequency, len(X))
        self.points_ = [category for category, count in counts.items() if count >= min_count]
        return self

    def transform(self, X) -> pd.Series:
        """Сводит значения к выученным категориям.

        Редкие и новые значения уходят в ``OTHER_BIN``, пропуски — в ``NAN_BIN``.

        Args:
            X: Фича для разметки.

        Returns:
            Упорядоченная категориальная Series (частые категории по частоте, затем other, missing).
        """
        self._check_fitted()
        X = as_1d(X).reset_index(drop=True).astype("object")

        out = X.where(X.isin(set(self.points_)), OTHER_BIN)  # редкие и новые значения -> other
        out[X.isna()] = NAN_BIN                              # пропуски -> missing

        out = pd.Series(
            pd.Categorical(out, categories=[*self.points_, OTHER_BIN, NAN_BIN], ordered=True)
        )
        self.bins_ = out.cat.categories
        return out

    def fit_transform(self, X) -> pd.Series:
        """Учит категории и сразу применяет их к тем же данным."""
        return self.fit(X).transform(X)

    def _check_fitted(self) -> None:
        if not hasattr(self, "points_"):
            raise RuntimeError("CatBinner не обучён: вызовите fit() перед transform().")
