
# Practika — Trash Detection (YOLOv8)

Проект по компьютерному зрению: обучение модели YOLOv8 для обнаружения мусора.

## Быстрый запуск (Windows)
1) Установи Python 3.11
2) Скачай/клонируй репозиторий
3) Запусти `run_realtime.bat` (двойной клик)

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

## Realtime (webcam)

### 1) Train model (or download weights)

```bash
python src/train.py
```

### 2) Run realtime webcam detection

```bash
python src/realtime.py
```

Press **Q** or **ESC** to quit (click on the video window first).

---

## Важно

Для полноценного запуска нужно:

1. Скачать датасет с Kaggle
2. Затем выполнить:

```bash
python src/coco_to_yolo.py
```

---

## 4) Закоммитить изменения (только код/README/.gitignore)

```bash
git add README.md .gitignore src
git commit -m "Add COCO conversion and realtime webcam detection"
git push
```
