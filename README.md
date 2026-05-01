# isp-assignment

Домашнее задание посвящено полному конвейеру обработки изображений. Вам предстоит обучить нейросетевые модели для блоков дебайеринга и баланса белого. Также обучить end-to-end пайплайн RAW-sRGB. 

Основной файл для всех результатов -- `main.ipynb`. 

## Данные

1. Скачайте архив с данными:   **https://www.kaggle.com/datasets/yasinowo/isp-assignment-dataset**
2. Распакуйте в корне репозитория

## Окружение и проверка

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```