# Предобработка Датасета Birds Milk

Набор скриптов для предобработки аудио датасета.

## Структура проекта

- `audio_separator.py` - Основной функционал для разделения и обработки аудио
- `task_convert_opus.py` - Обработка конвертации аудио в формат Opus
- `task_separate.py` и `task_separate2.py` - Реализации задачи разделения аудио

## Использование

### Конвертация аудио в формат Opus

```python
python task_convert_opus.py
```

Этот скрипт:
- Обрабатывает аудиофайлы из датасета Blackbird
- Конвертирует их в формат Opus с оптимизированными настройками
- Сохраняет метаданные при конвертации

### Отделение вокала

```python
python task_separate.py
```

Возможности:
- Извлекает вокал и убирает реверберацию из музыкальных треков

## Ссылки

- [Датасет Blackbird](https://github.com/Kiberchaika/The_Blackbird_Dataset)
- [Обучение разделения музыкальных источников](https://github.com/jarredou/Music-Source-Separation-Training) 