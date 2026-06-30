import sys

import nbformat as nbf

OUT = sys.argv[1] if len(sys.argv) > 1 else "binning_demo.ipynb"

nb = nbf.v4.new_notebook()
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells = []

cells.append(
    md(
        "# Демонстрация биннинга (`binning/`)\n"
        "\n"
        "Пакет `binning/` — два самостоятельных класса в стиле scikit-learn "
        "(`fit` / `transform`):\n"
        "\n"
        "- **`NumBinner`** — числовая фича: квантильные интервалы + «точки» "
        "(супервстречаемые значения) + слияние редких бинов;\n"
        "- **`CatBinner`** — категориальная фича: частые категории сохраняются, "
        "редкие → `other`, пропуски → `missing`.\n"
        "\n"
        "Биннинг **без таргета** (unsupervised) и ничего не знает про PSI — пригоден и для "
        "PSI, и для WoE. Тип фичи задаётся **явно** (выбором класса), а не по dtype.\n"
        "\n"
        "Данные генерируются прямо в ноутбуке (numpy) — примеры самодостаточны."
    )
)

cells.append(
    code(
        "import sys, pathlib\n"
        "sys.path.insert(0, str(pathlib.Path.cwd()))\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "from binning import NumBinner, CatBinner\n"
        "\n"
        "rng = np.random.default_rng(0)\n"
        "pd.set_option('display.width', 160)\n"
        "print('loaded')"
    )
)

cells.append(
    md(
        "## 1. `NumBinner` — числовой биннер\n"
        "\n"
        "Параметры конструктора:\n"
        "\n"
        "| Параметр | Назначение | По умолчанию |\n"
        "|---|---|---|\n"
        "| `n_bins` | число квантильных интервалов непрерывного фона | `10` |\n"
        "| `min_frequency` | минимальный размер бина; меньшие сливаются с соседом | `\"auto\"` |\n"
        "| `point_share` | порог доли, при котором значение становится «точкой» (спайк) | `0.10` |\n"
        "\n"
        "`fit(X)` учит границы, `transform(X)` размечает. Результат — упорядоченная "
        "`Categorical`-Series: интервал = `pd.Interval`, точка = само число, "
        "`other` / `missing` — служебные строки."
    )
)

cells.append(
    md(
        "### 1.1 `n_bins` — разрешение квантильной сетки\n"
        "\n"
        "Непрерывная фича делится на `n_bins` квантильных интервалов (примерно равных по "
        "числу наблюдений). Чем больше `n_bins`, тем мельче интервалы."
    )
)

cells.append(
    code(
        "x = pd.Series(rng.lognormal(mean=10, sigma=0.5, size=5000))\n"
        "for n in (4, 10):\n"
        "    cats = NumBinner(n_bins=n).fit_transform(x).cat.categories\n"
        "    print(f'n_bins={n:2d} -> {len(cats)} бинов')\n"
        "    print('   ', [str(c) for c in cats])"
    )
)

cells.append(
    md(
        "### 1.2 `point_share` — выделение «точек» (спайков)\n"
        "\n"
        "Если доля одного значения ≥ `point_share`, оно становится **отдельным бином-точкой**, "
        "а не растворяется в интервале. Нужно для супервстречаемых значений: ноль-инфляция, "
        "дефолтные заполнители, «магические» константы. Точка проходит ещё и порог "
        "`min_frequency` (бин не меньше минимального)."
    )
)

cells.append(
    code(
        "# 40% значений — ровно 0.0 (спайк), остальное непрерывно\n"
        "x = pd.Series(np.where(rng.random(5000) < 0.4, 0.0, rng.lognormal(8, 0.5, 5000)))\n"
        "for ps in (0.10, 0.50):\n"
        "    cats = list(NumBinner(point_share=ps).fit_transform(x).cat.categories)\n"
        "    is_point = any(isinstance(c, float) for c in cats)\n"
        "    print(f'point_share={ps:.2f} -> 0.0 отдельная точка? {is_point}  (спайк = 40%)')"
    )
)

cells.append(
    md(
        "### 1.3 `min_frequency` — порог редкого бина\n"
        "\n"
        "Бины меньше порога **сливаются** с соседним. Значения:\n"
        "\n"
        "- `\"auto\"` — `clip(0.01·N, 20, 1000)` (доля выборки с полом/потолком);\n"
        "- `float` из `(0, 1)` — доля от размера выборки;\n"
        "- `int ≥ 1` — абсолютное число наблюдений;\n"
        "- `None` / `0` — слияние выключено.\n"
        "\n"
        "Порог считается от **всей** обученной выборки и применяется единообразно ко всем бинам."
    )
)

cells.append(
    code(
        "x = pd.Series(rng.lognormal(10, 0.5, 3000))\n"
        "for mf in (None, 0.15, 0.30):\n"
        "    cats = NumBinner(n_bins=10, min_frequency=mf).fit_transform(x).cat.categories\n"
        "    intervals = [c for c in cats if isinstance(c, pd.Interval)]\n"
        "    print(f'min_frequency={str(mf):5} -> {len(intervals)} интервалов')"
    )
)

cells.append(
    md(
        "### 1.4 Малая кардинальность — дискретный режим\n"
        "\n"
        "Если уникальных значений ≤ `n_bins`, квантили не строятся: каждое частое значение "
        "получает свой бин (точку), редкие уйдут в `other` при `transform`. Так "
        "число-кодированные категории (например, 1..5 продуктов) не дробятся ошибочно."
    )
)

cells.append(
    code(
        "x = pd.Series(rng.choice([1, 2, 3, 4, 5], size=3000, p=[.4, .3, .18, .08, .04]))\n"
        "out = NumBinner(n_bins=10).fit_transform(x)\n"
        "print('бины:', list(out.cat.categories))\n"
        "out.value_counts().sort_index()"
    )
)

cells.append(
    md(
        "### 1.5 `fit` на train, `transform` на новых данных\n"
        "\n"
        "Границы учатся один раз на обучающей выборке и применяются к любым новым данным. "
        "Значения вне обученного диапазона попадают в крайние интервалы "
        "`(-inf, …]` / `(…, inf]`, не ломая разметку."
    )
)

cells.append(
    code(
        "binner = NumBinner(n_bins=5).fit(pd.Series(rng.lognormal(10, 0.4, 3000)))\n"
        "new = pd.Series([0.0, 1e9])  # значения за пределами train\n"
        "print(binner.transform(new).tolist())"
    )
)

cells.append(
    md(
        "## 2. `CatBinner` — категориальный биннер\n"
        "\n"
        "Единственный параметр — `min_frequency` (та же семантика порога). При `fit` "
        "сохраняются категории с числом наблюдений ≥ порога (по убыванию частоты). При "
        "`transform`:\n"
        "\n"
        "- редкие и **новые** (не виденные на `fit`) категории → `other`;\n"
        "- пропуски → `missing`."
    )
)

cells.append(
    code(
        "pool = ['web', 'mobile', 'branch', 'call_center', 'partner']\n"
        "pool += [f'rare_{i:02d}' for i in range(20)]\n"
        "p = [.35, .25, .18, .04, .03] + [.15 / 20] * 20\n"
        "x = pd.Series(rng.choice(pool, size=4000, p=p))\n"
        "\n"
        "for mf in ('auto', 0.05):\n"
        "    kept = CatBinner(min_frequency=mf).fit(x).points_\n"
        "    print(f'min_frequency={str(mf):5} -> сохранено {len(kept)}: {kept}')"
    )
)

cells.append(
    code(
        "# новая категория и пропуск на transform\n"
        "binner = CatBinner(min_frequency=0.05).fit(x)\n"
        "probe = pd.Series(['web', 'rare_03', 'BRAND_NEW', None])\n"
        "print(binner.transform(probe).tolist())"
    )
)

cells.append(
    md(
        "## 3. Структура результата\n"
        "\n"
        "`transform` возвращает упорядоченную `Categorical`-Series. Идентичность бина:\n"
        "\n"
        "- интервал → `pd.Interval` (right-closed `(a, b]`; у интервала, упёртого в точку, "
        "правая граница открыта);\n"
        "- точка → само число (`float`);\n"
        "- служебные → строки `other` / `missing`.\n"
        "\n"
        "Выученные атрибуты: `points_` (точки / частые категории), `edges_` (границы "
        "интервалов, у `NumBinner`), `bins_` (итоговые категории после `transform`). "
        "Результат напрямую идёт в расчёт PSI или WoE."
    )
)

cells.append(
    code(
        "b = NumBinner(n_bins=6).fit(pd.Series(rng.lognormal(10, 0.5, 3000)))\n"
        "print('points_:', b.points_)\n"
        "print('edges_ :', b.edges_)"
    )
)

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3.14 (.venv)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python"},
}

with open(OUT, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"{OUT} written: {len(cells)} cells")
