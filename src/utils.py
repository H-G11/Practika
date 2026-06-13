import os
import shutil
from pathlib import Path

def check_dataset(data_path):
    """Проверка структуры датасета"""
    path = Path(data_path)
    
    folders = ['train/images', 'train/labels', 
               'valid/images', 'valid/labels',
               'test/images', 'test/labels']
    
    for folder in folders:
        full_path = path / folder
        if full_path.exists():
            files = len(list(full_path.iterdir()))
            print(f"✅ {folder}: {files} файлов")
        else:
            print(f"❌ {folder}: папка не найдена!")

def get_model_info(model_path):
    """Информация о модели"""
    from ultralytics import YOLO
    model = YOLO(model_path)
    print(f"Модель загружена: {model_path}")
    return model

if __name__ == "__main__":
    check_dataset('./data')