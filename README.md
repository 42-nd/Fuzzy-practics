# Проект: Гибридный Fuzzy-Pipeline на примере Titanic

Данный репозиторий содержит учебный пример применения гибридного подхода к анализу данных с использованием нечеткой логики (библиотека [scikit-fuzzy](https://pythonhosted.org/scikit-fuzzy/))
В качестве демонстрационного датасета используется [Titanic](https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv)

## Содержание репозитория
```
fuzzy-practics/
├─ utils/
│   ├─ __init__.py
│   ├─ ModelAnalytic.py
├─ base_pipeline.ipynb
└─ README.md
```

- **`base_pipeline.ipynb`** – Основной Jupyter Notebook с примером гибридного подхода

## Как запустить

1. **Склонируйте репозиторий** или скачайте его содержимое любым удобным способом.
   ```bash
   git clone https://github.com/your_username/fuzzy-practics.git
   ```
2. **Установите необходимые библиотеки**.
   ```bash
   pip install -r requirements.txt
   ```


## Структура гибридного подхода
1. **Подготовка данных** (предобработка, фильтрация).
2. **Нечеткая логика** (fuzzy sets) для создания дополнительных признаков.
3. **Кластеризация** (KMeans) для выявления скрытых групп/шаблонов.


## Результаты
На примере датасета Titanic видно, что добавление нечетких признаков и кластера улучшает качество классификации (метрики F1, Precision, Recall). Нечеткая логика помогает выявить дополнительные закономерности в данных.


