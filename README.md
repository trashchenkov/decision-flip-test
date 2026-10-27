# decision-flip-test

Проверка моделей-решателей (decision models) на простой трюк из [разбора Arize](https://arize.com/blog/decision-model-benchmark/): задать один и тот же вопрос в двух противоположных формулировках и посмотреть, поменяется ли ответ.

Данные — [RAGTruth](https://github.com/ParticleMedia/RAGTruth) (MIT): ответы моделей по документам с ручной разметкой галлюцинаций. Берётся фиксированная случайная выборка из тестовой части (seed 42).

Вопросы:
- `A_en` — «Is every claim in the response supported by the context?»
- `B_en` — «Does the response contain any claim that is not supported by the context?»
- `IRR_fr` — «Is the response written in French?» (контроль: к галлюцинациям отношения не имеет)

Модель, которая читает вопрос, отвечает на A и B зеркально: корреляция её ответов близка к −1. Модель, которая вопрос игнорирует, отвечает на оба одинаково.

## Запуск

```bash
pip install scikit-learn
python prep.py                                  # скачать RAGTruth, собрать data/sample.json
ollama pull d3:4b                               # нужен Ollama 0.40.9+
python run.py ollama:d3:4b --limit 500 --out mine
python analyze.py                               # все файлы в results/
```

Свою модель можно добавить веткой в `load()` в `run.py`: функция получает `{"context", "response"}` и текст вопроса и возвращает вероятность «да».

`longctx.py` проверяет, что происходит со входом длиннее 4096 токенов в пакетах d3 для Ollama (ответ: HTTP 400 `limit is 4096`, текст молча не обрезается).

## Результаты (октябрь 2026)

| модель | AUC на A | AUC на B | corr(A, B) | n |
|---|---|---|---|---|
| d3:4b (Ollama) | 0.085 | 0.918 | −0.98 | 500 |
| d3:9b (Ollama) | 0.074 | 0.930 | −0.98 | 500 |
| Laya multilingual | 0.443 | 0.452 | +0.50 | 1000 |
| Strands Decider 2B | 0.227 | 0.220 | +0.92 | 1000 |

AUC считается по вероятности «да» как оценке галлюцинации, поэтому для вопроса A хороший результат — близко к 0, для B — близко к 1. Сырые ответы лежат в `results/`. d3 прогонялись на первых 500 примерах выборки, Laya и Strands — на всей 1000.

Код — MIT.
