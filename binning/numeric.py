"""Числовой биннинг: класс NumericBinner (fit/transform).

fit делит значения на точки (дискретные значения со своим бином) и квантильные интервалы:

    - мало уникальных (<= n_bins): дискретная фича — каждое ЧАСТОЕ значение = своя точка,
      редкие (< min_bin наблюдений) уходят в OTHER;
    - иначе: точками делаем супервстречаемые значения (доля >= point_share), остальное бьём
      на квантили; точки врезаются в границы, чтобы интервал их не накрывал.

Состояние после fit: points_ (точечные значения, float), edges_ (границы интервалов либо None
в дискретном режиме) и bins_ — упорядоченные ИДЕНТИЧНОСТИ бинов: интервал как pd.Interval,
точка как само число (float), служебные other/missing как строки. Класс ничего не знает про
PSI — годится и для WoE.
"""

from typing import Self

import numpy as np
import pandas as pd

from .common import NAN_BIN, OTHER_BIN, as_1d, resolve_min_count


class NumBinner:
    """Бьёт числовую фичу на бины. Учит границы на переданных данных (база/train)."""

    def __init__(self, *, n_bins: int = 10, min_bin: int | float | str | None = "auto",
                 point_share: float = 0.10):
        self.n_bins = n_bins
        self.min_bin = min_bin
        self.point_share = point_share

    def fit(self, X) -> Self:
        """Выучить бины: points_, edges_ (None -> дискретный) и идентичности bins_."""
        X = as_1d(X).dropna().to_numpy("float")

        # Проверка на пустой входной вектор
        if X.size == 0:
            self.points_, self.edges_ = (), np.array([-np.inf, np.inf])
            return self

        vals, counts = np.unique(X, return_counts=True)
        min_count = resolve_min_count(self.min_bin, X.size)

        # 1-й Этап (Низкая кардинальность): если уникальных <= n_bins — фича дискретная. Точками делаем только ЧАСТЫЕ
        # значения (>= min_count наблюдений); редкие значения уйдут в OTHER при transform.
        if vals.size <= self.n_bins:
            self.points_, self.edges_ = tuple(vals[counts >= min_count].tolist()), None
            return self

        # 2-ой Этап (Высокая кардинальность): ищем точки — супервстречаемые значения (доля >= point_share). Каждая точка — свой бин
        points = ()
        if self.point_share and 0.0 < self.point_share <= 1.0:
            points = tuple(vals[counts / X.size >= self.point_share].tolist())
        to_bin = X[~np.isin(X, points)] if points else X
        if to_bin.size == 0:                     # все значения — точки, интервалов нет
            self.points_, self.edges_ = points, None
            return self

        # 3-ий Этап: квантили по остатку, края -> ±inf, редкие интервалы соединяем, точки врезаем в границы
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
        """Применить бины. Интервал -> pd.Interval, точка -> само число, NaN -> NAN_BIN.

        В дискретном режиме (edges_=None) не-точечные/новые значения -> OTHER_BIN.
        Возвращает упорядоченную категориальную Series (категории = bins_).
        """
        self._check_fitted()
        X = as_1d(X).reset_index(drop=True).astype("float")

        # 1-й случай: Дискретная фича. Точки в бины - значения точек и бин other
        if self.edges_ is None:
            out = pd.Series(pd.Categorical(
                np.full(len(X), OTHER_BIN, dtype=object),
                categories=[OTHER_BIN],
                ordered=True
            ))
        # 2-случай: Непрерывная фича. Интервалы из edges_. Точки/missing - отдельные бины
        else:
            out = pd.cut(X, bins=self.edges_, include_lowest=True)

        # Разметка бинов точками
        for point in self.points_:   
            out = out.cat.add_categories(point)
            out[X == point] = point

        # Отдельный бин для пропусков
        out = out.cat.add_categories(NAN_BIN)
        out[X.isna()] = NAN_BIN

        # Сортируем категроии бинов
        out = self.sort_categories(out)
        self.bins_ = out.cat.categories

        return out

    def fit_transform(self, X) -> pd.Series:
        return self.fit(X).transform(X)

    
    @staticmethod
    def sort_categories(X: pd.Series) -> pd.Series:
        """Отсортировать категории в серии по «логике точек».

        Интервалы — по возрастанию; точку-число ставим сразу за «её» интервалом (тем, чья правая
        граница совпадает с точкой), и у этого интервала ОТКРЫВАЕМ правую границу (closed='neither');
        служебные other/missing — в конец. Значения границ и точек не меняем — только порядок и скобку.
        """
        categories_list = list(X.cat.categories)
        points = {category for category in categories_list if isinstance(category, float)}

        # идентичность: открыть правую границу у интервалов, упёртых в точку — по кодам строк (данные не теряются)
        X = X.cat.rename_categories(
            {interval: pd.Interval(interval.left, interval.right, closed="neither")
            for interval in categories_list if isinstance(interval, pd.Interval) and interval.right in points}
        )
        # порядок: интервал — по началу; точка — перед интервалом с тем же началом (т.е. ровно за своим)
        sorted_categories = sorted(
            (category for category in X.cat.categories if not isinstance(category, str)),
            key=lambda x: (x.left, 1) if isinstance(x, pd.Interval) else (x, 0)
        )
        extra_categories = [extra_cat for extra_cat in (OTHER_BIN, NAN_BIN) if extra_cat in categories_list]

        return X.cat.reorder_categories(sorted_categories + extra_categories, ordered=True)


    def _merge_rare(self, v: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Слить соседние интервалы, пока в каждом < min_bin наблюдений. Края ±inf не трогаем."""
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
            drop = i if i == counts.size - 1 else i + 1   # последний бин -> с левым, иначе -> с правым
            edges = np.delete(edges, drop)
        return edges

    def _check_fitted(self) -> None:
        if not hasattr(self, "edges_"):
            raise RuntimeError("NumericBinner не обучен: вызовите fit() перед transform().")
