from typing import Self

import numpy as np
import pandas as pd

from .common import NAN_BIN, OTHER_BIN, as_1d, resolve_min_count


class NumBinner:
    """Разбивает числовую фичу на точки и квантильные интервалы.

    Учит границы на переданных данных (база/train). Возможны два режима:

    - мало уникальных значений (<= n_bins) — дискретный режим: каждое частое значение
      получает свой бин, редкие (< min_bin наблюдений) уходят в other;
    - иначе — точками выделяются супервстречаемые значения (доля >= point_share), остальное
      бьётся на квантили; точки врезаются в границы, чтобы интервал их не накрывал.

    Идентичность бина: интервал — ``pd.Interval``, точка — само число (float), служебные
    other/missing — строки. Класс не знает про PSI и пригоден, например, для WoE.
    """

    def __init__(
            self,
            *,
            n_bins: int = 10,
            min_bin: int | float | str | None = "auto",
            point_share: float = 0.10
        ):
        self.n_bins = n_bins
        self.min_bin = min_bin
        self.point_share = point_share

    def fit(self, X) -> Self:
        """Учит точки points_, границы edges_ и идентичности бинов bins_ на данных.

        Args:
            X: Числовая фича (база/train).

        Returns:
            Сам обученный биннер.
        """
        X = as_1d(X).dropna().to_numpy("float")

        # Пустой вход: один интервал (-inf, inf), точек нет.
        if X.size == 0:
            self.points_, self.edges_ = (), np.array([-np.inf, np.inf])
            return self

        vals, counts = np.unique(X, return_counts=True)
        min_count = resolve_min_count(self.min_bin, X.size)

        # Низкая кардинальность (уникальных <= n_bins): дискретный режим. Точки — только частые
        # значения (>= min_count); редкие уйдут в other при transform.
        if vals.size <= self.n_bins:
            self.points_, self.edges_ = tuple(vals[counts >= min_count].tolist()), None
            return self

        # Высокая кардинальность: точки — супервстречаемые значения (доля >= point_share).
        points = ()
        if self.point_share and 0.0 < self.point_share <= 1.0:
            points = tuple(vals[counts / X.size >= self.point_share].tolist())
        to_bin = X[~np.isin(X, points)] if points else X
        if to_bin.size == 0:  # все значения — точки, интервалов нет
            self.points_, self.edges_ = points, None
            return self

        # Квантили по остатку: края -> ±inf, редкие интервалы сливаются, точки врезаются в границы.
        edges = np.unique(np.quantile(to_bin, np.linspace(0.0, 1.0, self.n_bins + 1)))
        if edges.size < 2:
            edges = np.array([-np.inf, np.inf])
        else:
            edges[0], edges[-1] = -np.inf, np.inf
            edges = self._merge_rare(to_bin, edges)
        if points:
            edges = np.unique(np.concatenate([edges, np.array(points, dtype="float")]))
        self.points_, self.edges_ = points, edges
        return self

    def transform(self, X) -> pd.Series:
        """Применяет выученные бины к фиче.

        Интервал размечается как ``pd.Interval``, точка — как само число, пропуск — как ``NAN_BIN``.
        В дискретном режиме (edges_ = None) не-точечные и новые значения уходят в ``OTHER_BIN``.

        Args:
            X: Фича для разметки.

        Returns:
            Упорядоченная категориальная Series (категории = bins_).
        """
        self._check_fitted()
        X = as_1d(X).reset_index(drop=True).astype("float")

        # Дискретная фича: стартуем со всех значений в other, точки разметятся ниже.
        if self.edges_ is None:
            out = pd.Series(pd.Categorical(
                np.full(len(X), OTHER_BIN, dtype=object),
                categories=[OTHER_BIN],
                ordered=True,
            ))
        # Непрерывная фича: интервалы из edges_.
        else:
            out = pd.cut(X, bins=self.edges_, include_lowest=True)

        # Разметка точечных бинов.
        for point in self.points_:
            out = out.cat.add_categories(point)
            out[X == point] = point

        # Отдельный бин для пропусков.
        out = out.cat.add_categories(NAN_BIN)
        out[X.isna()] = NAN_BIN

        out = self.sort_categories(out)
        self.bins_ = out.cat.categories
        return out

    def fit_transform(self, X) -> pd.Series:
        """Учит бины и сразу применяет их к тем же данным."""
        return self.fit(X).transform(X)

    @staticmethod
    def sort_categories(X: pd.Series) -> pd.Series:
        """Упорядочивает категории по «логике точек».

        Интервалы идут по возрастанию; точка-число ставится сразу за «своим» интервалом (тем,
        чья правая граница совпадает с точкой), и у этого интервала правая граница открывается
        (closed='neither'); служебные other/missing уходят в конец. Значения границ и точек не
        меняются — только порядок и тип скобки.

        Args:
            X: Категориальная Series бинов.

        Returns:
            Та же Series с упорядоченными категориями.
        """
        categories_list = list(X.cat.categories)
        points = {category for category in categories_list if isinstance(category, float)}

        # Открываем правую границу у интервалов, упёртых в точку (по кодам строк).
        X = X.cat.rename_categories(
            {interval: pd.Interval(interval.left, interval.right, closed="neither")
             for interval in categories_list
             if isinstance(interval, pd.Interval) and interval.right in points}
        )
        # Порядок: интервал — по левой границе; точка — перед интервалом с той же левой границей.
        sorted_categories = sorted(
            (category for category in X.cat.categories if not isinstance(category, str)),
            key=lambda x: (x.left, 1) if isinstance(x, pd.Interval) else (x, 0),
        )
        extra_categories = [extra for extra in (OTHER_BIN, NAN_BIN) if extra in categories_list]
        return X.cat.reorder_categories(sorted_categories + extra_categories, ordered=True)

    def _merge_rare(self, v: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Сливает соседние интервалы, пока в каждом < min_bin наблюдений; края ±inf не трогает."""
        min_count = resolve_min_count(self.min_bin, v.size)
        if min_count <= 0:
            return edges
        while edges.size > 2:
            codes = pd.cut(pd.Series(v), bins=edges, labels=False, include_lowest=True).to_numpy()
            counts = np.bincount(codes.astype(np.intp), minlength=edges.size - 1)
            small = np.where(counts < min_count)[0]
            if small.size == 0:
                break
            i = int(small[0])
            drop = i if i == counts.size - 1 else i + 1  # последний — с левым, иначе с правым
            edges = np.delete(edges, drop)
        return edges

    def _check_fitted(self) -> None:
        if not hasattr(self, "edges_"):
            raise RuntimeError("NumBinner не обучён: вызовите fit() перед transform().")
