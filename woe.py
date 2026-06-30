import numpy as np


def calc_woe_laplace(
    positive_events,
    total_positive_events,
    negative_events,
    total_negative_events,
    *,
    alpha: float = 0.5,
) -> np.ndarray:
    """Вычисляет WoE бинов со сглаживанием Лапласа.

    WoE бина = ln(доля_позитивов / доля_негативов),
    где доля = (events + alpha) / total_events.
    Добавление alpha к счётчикам (по умолчанию 0.5 — приор Джеффриса, как в PSI) убирает
    log(0)/деление на ноль в пустых бинах; alpha=0 — «сырая» формула (может дать ±inf).
    Положительный WoE — доля позитивов выше доли негативов, отрицательный — наоборот.
    Принимает скаляры или массивы (по элементу на бин).

    Args:
        positive_events: Число позитивов в бине (скаляр или массив по бинам).
        total_positive_events: Всего позитивов по всем бинам.
        negative_events: Число негативов в бине (скаляр или массив по бинам).
        total_negative_events: Всего негативов по всем бинам.
        alpha: Сглаживание Лапласа, добавляется к счётчикам бина.

    Returns:
        WoE по бинам (той же формы, что входные счётчики).
    """
    positive_rate = (
        np.asarray(positive_events, dtype=float) + alpha
    ) / total_positive_events
    negative_rate = (
        np.asarray(negative_events, dtype=float) + alpha
    ) / total_negative_events
    return np.log(positive_rate / negative_rate)


def calc_woe_epsilon(
    positive_events,
    total_positive_events,
    negative_events,
    total_negative_events,
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    """Вычисляет WoE бинов с epsilon-клиппингом.

    WoE бина = ln(доля_позитивов / доля_негативов), где доля = events / total_events,
    поднятая до eps (clip), чтобы убрать log(0)/деление на ноль. Классическая
    альтернатива сглаживанию Лапласа (``calc_woe_laplace``): пустой бин даёт не ±inf,
    а ограниченный крайний WoE ~ln(eps).
    Положительный WoE — доля позитивов выше доли негативов, отрицательный — наоборот.
    Принимает скаляры или массивы (по элементу на бин).

    Args:
        positive_events: Число позитивов в бине (скаляр или массив по бинам).
        total_positive_events: Всего позитивов по всем бинам.
        negative_events: Число негативов в бине (скаляр или массив по бинам).
        total_negative_events: Всего негативов по всем бинам.
        eps: Нижняя граница доли.

    Returns:
        WoE по бинам (той же формы, что входные счётчики).
    """
    positive_rate = np.asarray(positive_events, dtype=float) / total_positive_events
    negative_rate = np.asarray(negative_events, dtype=float) / total_negative_events
    return np.log(np.clip(positive_rate, eps, None) / np.clip(negative_rate, eps, None))
