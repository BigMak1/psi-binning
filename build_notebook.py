import sys

import nbformat as nbf

OUT = sys.argv[1] if len(sys.argv) > 1 else "psi_playground.ipynb"

nb = nbf.v4.new_notebook()
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells = []

cells.append(md(
    "# PSI playground\n"
    "\n"
    "Песочница для бининга фичей (PSI / WoE) и мониторинга стабильности.\n"
    "\n"
    "| Файл | Назначение |\n"
    "|------|------------|\n"
    "| `synthetic_data.py` | генерация выборки (12k строк, 12 месяцев, 7 фичей) |\n"
    "| `binning/` | пакет: `NumBinner` / `CatBinner` (fit/transform), независим от PSI |\n"
    "| `psi.py` | `define_base_data` (выбор базы) + `calc_psi_by_features` / `calc_psi_by_period` |\n"
    "| `psi_plots.py` | графики plotly (распределения линиями + PSI) |\n"
    "\n"
    "Запусти ячейки сверху вниз. Kernel — **Python 3.14 (.venv)**."
))

cells.append(code(
    "import sys, pathlib, importlib\n"
    "sys.path.insert(0, str(pathlib.Path.cwd()))\n"
    "\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "\n"
    "import synthetic_data, psi, psi_plots\n"
    "from binning import NumBinner, CatBinner\n"
    "for m in (synthetic_data, psi, psi_plots):\n"
    "    importlib.reload(m)\n"
    "\n"
    "# тип фичи задаётся ЯВНО (не по префиксу/dtype)\n"
    "NUMERIC = ['f__income', 'f__score', 'f__txn_count', 'f__rate', 'f__num_products']\n"
    "CATEGORICAL = ['c__region', 'c__channel']\n"
    "FEATURES = NUMERIC + CATEGORICAL\n"
    "\n"
    "pd.set_option('display.width', 160)\n"
    "pd.set_option('display.max_columns', 30)\n"
    "print('modules loaded')"
))

cells.append(md("## 1. Данные"))

cells.append(code(
    "from pathlib import Path\n"
    "DATA = Path('data/sample.parquet')\n"
    "if not DATA.exists():\n"
    "    synthetic_data.save_sample(synthetic_data.generate_sample(n_total=12_000, seed=42), str(DATA))\n"
    "df = synthetic_data.load_sample(str(DATA))\n"
    "print('shape:', df.shape)\n"
    "df.head()"
))

cells.append(code("df['sample_month'].value_counts().sort_index()"))

cells.append(md(
    "## 2. Профиль фичей и краевые случаи\n"
    "Каждая фича сконструирована под свой краевой случай бининга."
))

cells.append(code("df[synthetic_data.FEATURES_NUMERIC].describe()"))

cells.append(code(
    "print('f__income    доля NaN   =', round(df['f__income'].isna().mean(), 3))\n"
    "print('f__txn_count  доля нулей =', round((df['f__txn_count'] == 0).mean(), 3))\n"
    "print('f__rate       доля 0.5   =', round((df['f__rate'] == 0.5).mean(), 3))\n"
    "print('f__num_products уникальные =', sorted(df['f__num_products'].unique().tolist()))\n"
    "print('c__channel    уник. кат. =', df['c__channel'].nunique())"
))

cells.append(md(
    "## 3. Биннеры — fit / transform (sklearn-стиль)\n"
    "`NumBinner` (числовой) и `CatBinner` (категориальный) — самостоятельные классы. Учим бины на базе, "
    "применяем ко всем данным. Бининг ничего не знает про PSI — годится и для WoE."
))

cells.append(code(
    "# числовой: спайк нулей -> точечный бин, интервалы не пересекают точку\n"
    "NumBinner(n_bins=10).fit_transform(df['f__txn_count']).value_counts().sort_index()"
))

cells.append(code(
    "# f__rate: спайк 0.5 В СЕРЕДИНЕ; f__num_products: мало уникальных -> дискретный режим\n"
    "print('rate :', list(NumBinner().fit_transform(df['f__rate']).cat.categories))\n"
    "print('num_products :', list(NumBinner().fit_transform(df['f__num_products']).cat.categories))"
))

cells.append(code(
    "# тип фичи выбираем ЯВНО (не по dtype): число-кодированная категория\n"
    "codes = pd.Series(np.random.default_rng(0).integers(1000, 1030, size=len(df)))\n"
    "print('NumBinner ->', len(NumBinner().fit_transform(codes).cat.categories),\n"
    "      'бинов (квантили — неверно)')\n"
    "print('CatBinner ->', len(CatBinner().fit_transform(codes).cat.categories),\n"
    "      'бинов (категории — верно)')"
))

cells.append(md(
    "## 4. Выбор базы (reference)\n"
    "`psi.define_base_data` (переехала из удалённого reference.py): `base_size` (доля), "
    "`shift_size` (сдвиг окна), `mask_col` (готовая разметка). `None` -> вся выборка."
))

cells.append(code(
    "for kw in [{}, {'base_size': 0.2}, {'base_size': 0.2, 'shift_size': 0.2}]:\n"
    "    m = psi.define_base_data(df, **kw)\n"
    "    print(f'{str(kw):45s} -> {int(m.sum()):5d} строк')"
))

cells.append(md(
    "## 5. Распределения бинов и PSI по периодам\n"
    "База — первые 20% по дате (`psi_base_size=0.2`). Тип фичи — через `is_category`. "
    "Сверху доли бинов (линии), снизу PSI с порогами `0.10` / `0.25`."
))

cells.append(code(
    "CAT = set(CATEGORICAL)\n"
    "for feat in FEATURES:\n"
    "    psi_plots.plot_feature(df, feat, is_category=(feat in CAT), psi_base_size=0.2).show()"
))

cells.append(md("## 6. Сводная таблица PSI по периодам"))

cells.append(code(
    "psi_tbl = psi.calc_psi_by_features(df, FEATURES, cat_features=CATEGORICAL, psi_base_size=0.2)\n"
    "\n"
    "def _hl(v):\n"
    "    if v >= psi_plots.PSI_ALERT:\n"
    "        return 'background-color:#f8b4b4'  # >=0.25 alert\n"
    "    if v >= psi_plots.PSI_WARN:\n"
    "        return 'background-color:#fde68a'  # >=0.10 warn\n"
    "    return ''\n"
    "\n"
    "psi_tbl.round(4).style.map(_hl)"
))

cells.append(md(
    "## 7. Архитектура\n"
    "\n"
    "- **`binning/`** — `NumBinner` / `CatBinner` (fit/transform), тип выбирается ЯВНО, независим от PSI → "
    "переиспользуем для WoE: `NumBinner().fit(train); .transform(any)`.\n"
    "- **`psi.py`** — `define_base_data` (выбор базы: `base_size` / `shift_size` / `mask_col`) + "
    "`calc_psi_by_features` / `calc_psi_by_period`; параметры с префиксами `binner_*` / `psi_base_*`.\n"
    "- **Краевые случаи:** спайки супервстречаемых значений (`= v`, интервалы не пересекают точку), "
    "дискретный режим (мало уникальных), слияние редких бинов (`min_bin=\"auto\"`), Laplace-сглаживание."
))

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3.14 (.venv)", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

with open(OUT, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"{OUT} written: {len(cells)} cells")
