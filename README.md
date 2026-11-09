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
| `gemma4_no_thinking.jinja` | Переносимый Gemma-4 chat template для рабочего GGUF |
| `original-chat-template.jinja` | Исходный 3046-byte template из pre-patch GGUF для точного rollback |
| `patch_gguf_chat_template_inplace.py` | Metadata-only замена/откат `tokenizer.chat_template` без движения тензоров |

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

## GGUF Q8_0

Для E1 использовался актуальный clone `/home/dgx-alex/llama.cpp`. Конвертер создал Q8_0 **непосредственно** из merged BF16; промежуточный BF16/F16 GGUF и `llama-quantize` в этом запуске не использовались.

```bash
mkdir -p /home/dgx-alex/Models/Eva-Gemma4-31B-E1-GGUF

docker run --rm \
  --name eva-gemma4-e1-q8-convert \
  -v /home/dgx-alex:/home/dgx-alex \
  -w /home/dgx-alex/llama.cpp \
  unsloth-dgx-spark:25.11 \
  python3 convert_hf_to_gguf.py \
    /home/dgx-alex/Models/Eva-Gemma4-31B-E1-20261106-Merged-BF16 \
    --outfile /home/dgx-alex/Models/Eva-Gemma4-31B-E1-GGUF/Eva-Gemma4-31B-E1-Q8_0.partial.gguf \
    --outtype q8_0 \
    --model-name Eva-Gemma4-31B-E1

mv \
  /home/dgx-alex/Models/Eva-Gemma4-31B-E1-GGUF/Eva-Gemma4-31B-E1-Q8_0.partial.gguf \
  /home/dgx-alex/Models/Eva-Gemma4-31B-E1-GGUF/Eva-Gemma4-31B-E1-Q8_0.gguf
```

## Vision projector

Для Gemma-4 vision нужен отдельный matching `mmproj`, созданный из той же merged BF16 lineage:

```bash
docker run --rm \
  --name eva-gemma4-e1-mmproj-convert \
  -v /home/dgx-alex:/home/dgx-alex \
  -w /home/dgx-alex/llama.cpp \
  unsloth-dgx-spark:25.11 \
  python3 convert_hf_to_gguf.py \
    /home/dgx-alex/Models/Eva-Gemma4-31B-E1-20261106-Merged-BF16 \
    --outfile /home/dgx-alex/Models/Eva-Gemma4-31B-E1-GGUF/mmproj-Eva-Gemma4-31B-E1-BF16.partial.gguf \
    --outtype bf16 \
    --model-name Eva-Gemma4-31B-E1-mmproj \
    --mmproj

mv \
  /home/dgx-alex/Models/Eva-Gemma4-31B-E1-GGUF/mmproj-Eva-Gemma4-31B-E1-BF16.partial.gguf \
  /home/dgx-alex/Models/Eva-Gemma4-31B-E1-GGUF/mmproj-Eva-Gemma4-31B-E1-BF16.gguf
```

## Metadata-only chat-template patch

Обычный ответ попадал в Thinking из-за сочетания встроенного Gemma-4 template/channel routing и reasoning parser frontend. Повторная конвертация или квантизация не нужна: был изменён только metadata key `tokenizer.chat_template`.

Патчер сохраняет исходный шаблон, дополняет более короткий Jinja-комментарием до прежней длины, меняет байты **in-place** и проверяет read-back. Размер GGUF и tensor offsets остаются прежними.

Сначала скопировать три файла этого репозитория в рабочий каталог DGX:

```text
/home/dgx-alex/eva_gemma4_sft/patch_gguf_chat_template_inplace.py
/home/dgx-alex/eva_gemma4_sft/gemma4_no_thinking.jinja
/home/dgx-alex/eva_gemma4_sft/original-chat-template.jinja
```

`original-chat-template.jinja` в репозитории — проверенная копия исходной metadata (SHA-256 `cf661f74b5ec5a1a245ded9b988598dd73d5ab9cdb6872943d7d4e89c0bb97b6`). При первом patch скрипт также самостоятельно сохраняет template непосредственно из конвертированного GGUF.

Затем проверить исходный hash и выполнить patch:

```bash
sha256sum /home/dgx-alex/Models/Eva-Gemma4-31B-E1-GGUF/Eva-Gemma4-31B-E1-Q8_0.gguf

docker run --rm \
  -v /home/dgx-alex:/home/dgx-alex \
  -w /home/dgx-alex/llama.cpp \
  unsloth-dgx-spark:25.11 \
  python3 /home/dgx-alex/eva_gemma4_sft/patch_gguf_chat_template_inplace.py \
    /home/dgx-alex/Models/Eva-Gemma4-31B-E1-GGUF/Eva-Gemma4-31B-E1-Q8_0.gguf \
    /home/dgx-alex/eva_gemma4_sft/gemma4_no_thinking.jinja \
    /home/dgx-alex/Models/Eva-Gemma4-31B-E1-GGUF/original-chat-template.jinja

sha256sum /home/dgx-alex/Models/Eva-Gemma4-31B-E1-GGUF/Eva-Gemma4-31B-E1-Q8_0.gguf
```

Фактический результат E1:

```text
GGUF size до/после:  32,635,660,960 bytes
template до:         3046 bytes
новый Jinja:         1159 bytes
metadata после pad:  3046 bytes

Q8 pre-patch SHA-256:
f67ab91f202f19055511e2a7764c17cf3d1731ad7c947a377c76ad8e2016c2a5

Q8 post-patch SHA-256:
80abf69a94ca8a82da8dc86c3306003682394a452171ede8100eae3dcdf56929

mmproj BF16 SHA-256:
c51a2a40cc18662931eef666974a58e33535d5570d47fee2ba93c2857595db49
```

Smoke после patch выполнялся **без специальных reasoning flags**:

```bash
/home/dgx-alex/llama.cpp/build/bin/llama-cli \
  -m /home/dgx-alex/Models/Eva-Gemma4-31B-E1-GGUF/Eva-Gemma4-31B-E1-Q8_0.gguf \
  -c 1024 -n 32 --temp 0 --seed 3407 \
  --no-warmup --no-display-prompt --no-show-timings \
  -st -sys "I'm Eva" \
  -p "Ответь одним словом: ты здесь?"
```

Ожидаемый ответ: `Да.` без Thinking-блока.

### Откат metadata patch

Файл `original-chat-template.jinja` обязателен для отката. Улучшенный патчер поддерживает равную длину, поэтому исходный template можно вернуть тем же способом; текущая patched metadata при этом сохранится отдельно:

```bash
docker run --rm \
  -v /home/dgx-alex:/home/dgx-alex \
  -w /home/dgx-alex/llama.cpp \
  unsloth-dgx-spark:25.11 \
  python3 /home/dgx-alex/eva_gemma4_sft/patch_gguf_chat_template_inplace.py \
    /home/dgx-alex/Models/Eva-Gemma4-31B-E1-GGUF/Eva-Gemma4-31B-E1-Q8_0.gguf \
    /home/dgx-alex/eva_gemma4_sft/original-chat-template.jinja \
    /home/dgx-alex/Models/Eva-Gemma4-31B-E1-GGUF/patched-chat-template.jinja
```

После отката пересчитать SHA-256 и повторить smoke. При точном восстановлении metadata hash Q8 должен вернуться к pre-patch значению.

Модель, датасет, checkpoints и веса в репозиторий не входят.
