# Gemma-4 31B SFT up to 8K

Проверенный набор скриптов для QLoRA/SFT модели **Gemma-4-31B-it** с контекстом до **8192 токенов**. Рецепт использовался на NVIDIA DGX Spark (GB10, ARM64) в Docker-образе `unsloth-dgx-spark:25.11`.

## Что лежит в репозитории

| Файл | Назначение |
|---|---|
| `train_gemma4_31b_dgx.py` | Загрузка JSONL, audit, QLoRA-обучение и сохранение `final_lora` |
| `structural_mask.py` | Структурная assistant-only маска: loss только на всех ответах assistant |
| `mask_audit.py` | Проверка, что system/user/padding не попали в labels |
| `merge_gemma4_e1_bf16.py` | Безопасный merge `BF16 base + LoRA → merged BF16` |
| `audit_gemma4_e1_merged.py` | Проверка структуры, dtype и provenance merged-модели |
| `parity_gemma4_e1_bf16.py` | Сравнение logits/generation у `base+LoRA` и merged BF16 |

## Датасет

Поддерживается JSONL с одним диалогом на строку в любом из двух форматов:

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

или ShareGPT-подобный `conversations` с ролями `human/gpt/system`.

## Основные defaults

```text
context:          8192
LoRA rank/alpha:  32 / 64
epochs:           1
microbatch:       1
grad accumulation:16
learning rate:    1e-4
warmup steps:     28
optimizer:        adamw_8bit
```

Примеры длиннее контекста по умолчанию удаляются целиком, а не обрезаются.

## Запуск на DGX Spark

Пути можно изменить в начале trainer или передать через аргументы:

```bash
docker run --rm -it \
  --gpus=all \
  --ipc=host \
  --ulimit memlock=-1 \
  -v /home/dgx-alex:/home/dgx-alex \
  -w /home/dgx-alex/eva_gemma4_sft \
  unsloth-dgx-spark:25.11 \
  python3 train_gemma4_31b_dgx.py \
    --model /home/dgx-alex/Models/unsloth-gemma-4-31B-it-unsloth-bnb-4bit \
    --data /home/dgx-alex/eva_unsloth_core/dataset.jsonl \
    --out /home/dgx-alex/Models/Gemma4-31B-SFT-E1
```

Перед настоящим обучением скрипт печатает effective configuration и ждёт `Enter`. Для короткой проверки pipeline добавь `--max-steps 2`; для продолжения из последнего checkpoint — `--resume`.

## Результат

В каталоге `--out` появятся:

```text
checkpoint-*/
run_metrics.json
final_lora/
```

`final_lora/` — итоговый PEFT/LoRA adapter, а не полная модель.

## Merge в BF16

Перед запуском проверь константы `BASE`, `ADAPTER` и `FINAL` в трёх merge/audit/parity-скриптах. Для merge нужен **полный BF16 base**; 4-bit training checkpoint нельзя использовать как master source.

```bash
python3 merge_gemma4_e1_bf16.py
python3 audit_gemma4_e1_merged.py
python3 parity_gemma4_e1_bf16.py
```

Скрипт merge не изменяет base и adapter, пишет сначала в `.partial` и отказывается перезаписывать существующий итоговый каталог.

Модель, датасет, checkpoints и веса в репозиторий не входят.
