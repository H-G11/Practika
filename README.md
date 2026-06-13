# Practika — Trash Detection (YOLOv8)

Проект по компьютерному зрению: обучение модели YOLOv8 для обнаружения мусора.

## Установка

```bash
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
```

## Датасет (Kaggle)

Датасет не хранится в репозитории из-за размера.

Ссылка:  
https://www.kaggle.com/datasets/kneroma/tacotrashdataset

Скачивание:

```bash
kaggle datasets download -d kneroma/tacotrashdataset -p data --unzip
```

## Подготовка датасета

```bash
python src/prepare_dataset.py
```

## Обучение

```bash
python src/train.py
```

## Инференс (детекция)

Без аргументов:

```bash
python src/detect.py
```

Или конкретный файл:

```bash
python src/detect.py data/test/images/000004.jpg
```

Результаты сохраняются в `results/predict/`.