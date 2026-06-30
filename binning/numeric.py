from collections.abc import Sequence
from typing import Self

import numpy as np
import pandas as pd

from .common import NAN_BIN, OTHER_BIN, as_1d, resolve_min_count


class NumBinner:
    """Разбивает числовую фичу на точки и квантильные интервалы.

    Учит границы на переданных данных (база/train). Возможны два режима:

    - мало уникальных значений (<= n_bins) — дискретный режим: каждое частое значение
      получает свой бин, редкие (< min_frequency наблюдений) уходят в other;
    - иначе — точками выделяются супервстречаемые значения (доля >= point_share), остальное
      бьётся на квантили; точки врезаются в границы, чтобы интервал их не накрывал.

    Идентичность бина: интервал — ``pd.Interval``, точка — само число (float), служебные
    other/missing — строки. Класс не знает про PSI и пригоден, например, для WoE.
    """

    def __init__(
            self,
            *,
            n_bins: int = 10,
            min_frequency: int | float | str | None = "auto",
            point_share: float = 0.10
        ):
        self.n_bins = n_bins
        self.min_frequency = min_frequency
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
            self.points_, self.edges_ = [], np.array([-np.inf, np.inf])
            return self

        vals, counts = np.unique(X, return_counts=True)
        min_count = resolve_min_count(self.min_frequency, X.size)

        # Низкая кардинальность (уникальных <= n_bins): дискретный режим. Точки — только частые
        # значения (>= min_count); редкие уйдут в other при transform.
        if vals.size <= self.n_bins:
            self.points_, self.edges_ = vals[counts >= min_count].tolist(), None
            return self

        # Высокая кардинальность: точка — супервстречаемое значение, проходящее ОБА порога:
        # долю point_share (это спайк) и абсолютный min_count (бин не меньше min_frequency).
        points = []
        if self.point_share and 0.0 < self.point_share <= 1.0:
            point_floor = max(round(self.point_share * X.size), min_count)
            points = vals[counts >= point_floor].tolist()
        to_bin = X[~np.isin(X, points)] if points else X
        if to_bin.size == 0:  # все значения — точки, интервалов нет
            self.points_, self.edges_ = points, None
            return self

        # Границы: внутренние квантили остатка + точки-разделители, края — ±inf. np.unique
        # сортирует и убирает совпадения (точку, попавшую ровно на квантиль). Затем редкие
        # интервалы сливаются ПОСЛЕ врезки точек (точки и ±inf не удаляются), чтобы порог
        # min_count действовал и на «склоны» у точек, а каждый бин был >= min_frequency.
        quantiles = np.quantile(to_bin, np.linspace(0.0, 1.0, self.n_bins + 1))[1:-1]
        edges = np.unique(np.concatenate([[-np.inf], quantiles, points, [np.inf]]))
        self.points_, self.edges_ = points, self._merge_rare(to_bin, edges, min_count, points)
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

    @staticmethod
    def _merge_rare(
        X: np.ndarray,
        edges: np.ndarray,
        min_count: int,
        protected: Sequence[float] = ()
    ) -> np.ndarray:
        """Сливает соседние интервалы, пока в каждом < min_count наблюдений.

        Порог ``min_count`` считается один раз от полной базы (см. fit) и передаётся готовым,
        чтобы доля min_frequency означала долю всей выборки, а не остатка без точек. Защищённые
        границы — точки и края ±inf — при слиянии не удаляются: тонкий «склон» у точки
        сливается с внешним соседом, сама точка остаётся отдельным бином.

        Args:
            X: Непрерывные наблюдения (без точек), по которым считаются размеры интервалов.
            edges: Границы интервалов, включая ±inf и врезанные точки.
            min_count: Минимальный размер интервала в наблюдениях.
            protected: Границы-точки; края ±inf защищаются автоматически.

        Returns:
            Прореженные границы интервалов.
        """
        if min_count <= 0:
            return edges
        keep = set(protected) | {-np.inf, np.inf}
        while edges.size > 2:
            X_binned = pd.cut(pd.Series(X), bins=edges, include_lowest=True)
            X_counts = X_binned.value_counts(sort=False).to_numpy()
            for i in map(int, np.where(X_counts < min_count)[0]):
                # Сливаем интервал, удаляя одну его непрерывную границу: правую, иначе левую.
                if edges[i + 1] not in keep:
                    drop = i + 1
                elif edges[i] not in keep:
                    drop = i
                else:
                    continue  # обе границы защищены (точки/±inf) — интервал не сливаем
                edges = np.delete(edges, drop)
                break
            else:
                break  # маленьких интервалов, которые можно слить, не осталось
        return edges

    def _check_fitted(self) -> None:
        if not hasattr(self, "edges_"):
            raise RuntimeError("NumBinner не обучён: вызовите fit() перед transform().")
