from __future__ import annotations

import calendar
import datetime as dt
import os

import numpy as np
import pandas as pd

# 12 календарных месяцев — период наблюдения.
MONTHS = [dt.date(2024, m, 1) for m in range(1, 13)]

FEATURES_NUMERIC = ["f__income", "f__score", "f__txn_count", "f__rate", "f__num_products"]
FEATURES_CATEGORICAL = ["c__region", "c__channel"]
FEATURES = FEATURES_NUMERIC + FEATURES_CATEGORICAL


def _month_sizes(rng: np.random.Generator, n_total: int, n_months: int) -> np.ndarray:
    """Возвращает размеры месяцев: растущий тренд и шум, в сумме ровно ``n_total``."""
    trend = np.linspace(1.0, 1.3, n_months)  # +30% объёма за год
    noise = rng.normal(1.0, 0.08, n_months).clip(0.5, 1.5)
    weights = trend * noise
    probs = weights / weights.sum()
    return rng.multinomial(n_total, probs)


def _dates_in_month(rng: np.random.Generator, month_first: dt.date, n: int) -> list:
    """Возвращает случайные даты внутри месяца (равномерно по дням)."""
    n_days = calendar.monthrange(month_first.year, month_first.month)[1]
    days = rng.integers(1, n_days + 1, size=n)
    return [dt.date(month_first.year, month_first.month, int(d)) for d in days]


# Генераторы признаков (t — индекс месяца, 0..11). Каждый признак моделирует свой краевой случай.
def _gen_income(rng: np.random.Generator, n: int, t: int) -> np.ndarray:
    """Генерирует доход: лог-нормаль с дрейфом медианы (~50k -> ~66k), длинный хвост, ~4% NaN."""
    mu = np.log(50_000) + (np.log(66_000) - np.log(50_000)) * (t / 11)
    x = rng.lognormal(mean=mu, sigma=0.55, size=n)
    x[rng.random(n) < 0.04] = np.nan  # пропуски
    return x


def _gen_score(rng: np.random.Generator, n: int, t: int) -> np.ndarray:
    """Генерирует стабильную величину в [0, 1] без дрейфа — отрицательный контроль (PSI ~ 0)."""
    return rng.beta(2.5, 2.5, size=n)


def _gen_txn_count(rng: np.random.Generator, n: int, t: int) -> np.ndarray:
    """Генерирует счётчик транзакций: спайк нулей (~55%) и пуассоновский хвост."""
    p_zero = 0.58 - 0.006 * t            # доля нулей слегка падает: 0.58 -> ~0.51
    out = 1 + rng.poisson(lam=3.0, size=n)  # ненулевой хвост: 1, 2, 3, ...
    out[rng.random(n) < p_zero] = 0      # спайк нулей
    return out.astype(np.int64)


def _gen_region(rng: np.random.Generator, n: int, t: int) -> np.ndarray:
    """Генерирует регион: 5 стабильных категорий (низкая кардинальность)."""
    cats = ["North", "South", "East", "West", "Central"]
    probs = [0.30, 0.25, 0.20, 0.15, 0.10]
    return rng.choice(cats, size=n, p=probs)


def _channel_weights(t: int) -> dict:
    """Возвращает веса каналов месяца ``t``: дрейф mobile/branch, редкий хвост, новая категория."""
    w = {
        "web": 0.30,
        "mobile": 0.18 + 0.015 * t,  # растёт
        "branch": 0.24 - 0.015 * t,  # падает
        "call_center": 0.10,
        "partner": 0.08,
    }
    for i in range(20):  # редкий длинный хвост (высокая кардинальность)
        w[f"aggregator_{i:02d}"] = 0.10 / 20
    if t >= 6:  # новая категория с 7-го месяца
        w["super_app"] = 0.07
    s = sum(w.values())
    return {k: v / s for k, v in w.items()}


def _gen_channel(rng: np.random.Generator, n: int, t: int) -> np.ndarray:
    """Генерирует канал: высокая кардинальность, редкий хвост и новая категория со временем."""
    w = _channel_weights(t)
    cats = list(w.keys())
    probs = np.array(list(w.values()), dtype=float)
    probs = probs / probs.sum()
    return rng.choice(cats, size=n, p=probs)


def _gen_rate(rng: np.random.Generator, n: int, t: int) -> np.ndarray:
    """Генерирует ставку: спайк 0.5 в середине распределения (~35%) и непрерывный фон [0, 1]."""
    x = rng.beta(2.5, 2.5, size=n)  # непрерывный фон вокруг 0.5
    x[rng.random(n) < 0.35] = 0.5   # супервстречаемое значение в середине
    return x


def _gen_num_products(rng: np.random.Generator, n: int, t: int) -> np.ndarray:
    """Генерирует число продуктов 1..5 (мало уникальных -> дискретный режим) с дрейфом вправо."""
    base = np.array([0.40, 0.30, 0.18, 0.08, 0.04])
    shift = 0.012 * t
    p = base + np.array([-shift, -shift / 2, 0.0, shift / 2, shift])
    p = np.clip(p, 0.01, None)
    p = p / p.sum()
    return rng.choice([1, 2, 3, 4, 5], size=n, p=p).astype(np.int64)


def generate_sample(n_total: int = 12_000, seed: int = 42) -> pd.DataFrame:
    """Генерирует синтетическую выборку за 12 месяцев.

    Формат повторяет продовый: ``sample_date_orig`` (дата), ``sample_month`` (строка-якорь),
    числовые ``f__*`` и категориальные ``c__*`` признаки. Каждый признак моделирует свой
    краевой случай бининга.

    Args:
        n_total: Примерное число строк во всей выборке.
        seed: Сид генератора случайных чисел.

    Returns:
        DataFrame выборки.
    """
    rng = np.random.default_rng(seed)
    sizes = _month_sizes(rng, n_total, len(MONTHS))

    parts = []
    for t, (month_first, n) in enumerate(zip(MONTHS, sizes)):
        n = int(n)
        if n == 0:
            continue
        part = pd.DataFrame(
            {
                "sample_date_orig": _dates_in_month(rng, month_first, n),
                "sample_month": month_first.strftime("%Y-%m-%d"),
                "f__income": _gen_income(rng, n, t),
                "f__score": _gen_score(rng, n, t),
                "f__txn_count": _gen_txn_count(rng, n, t),
                "f__rate": _gen_rate(rng, n, t),
                "f__num_products": _gen_num_products(rng, n, t),
                "c__region": _gen_region(rng, n, t),
                "c__channel": _gen_channel(rng, n, t),
            }
        )
        parts.append(part)

    return pd.concat(parts, ignore_index=True)


def save_sample(df: pd.DataFrame, path: str = "data/sample.parquet") -> None:
    """Сохраняет выборку в parquet (типы сохраняются).

    Args:
        df: Выборка для сохранения.
        path: Путь к parquet-файлу.
    """
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    df.to_parquet(path, index=False)


def load_sample(path: str = "data/sample.parquet") -> pd.DataFrame:
    """Загружает выборку, приводя форматы (sample_date_orig -> date, sample_month -> str).

    Args:
        path: Путь к parquet-файлу.

    Returns:
        DataFrame выборки.
    """
    df = pd.read_parquet(path)
    df["sample_date_orig"] = pd.to_datetime(df["sample_date_orig"]).dt.date
    df["sample_month"] = df["sample_month"].astype(str)
    return df


if __name__ == "__main__":
    df = generate_sample()
    save_sample(df)
    period = f"{df['sample_month'].min()} .. {df['sample_month'].max()}"
    print(f"Сгенерировано {len(df)} строк; период {period}")
    print(df.head())
