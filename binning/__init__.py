"""Биннинг фичей (числовых и категориальных) в стиле fit/transform.

Классы (самостоятельные fit/transform, переиспользуемы вне PSI — напр. для WoE):
    NumBinner          — числовой биннинг (binning.numeric)
    CatBinner          — категориальный биннинг (binning.categorical)

    NAN_BIN, OTHER_BIN — метки служебных бинов

Тип фичи выбирает вызывающий код явно (без авто-инференса по dtype): число-кодированная
категория иначе ошибочно бьётся по квантилям.
"""

from .categorical import CatBinner
from .common import NAN_BIN, OTHER_BIN
from .numeric import NumBinner

__all__ = [
    "NumBinner",
    "CatBinner",
    "NAN_BIN",
    "OTHER_BIN",
]
