# ModelLab

Fine-tune a small language model on one narrow task three ways (full fine-tuning, LoRA, QLoRA), compare them with prompting baselines, and measure what each one costs.

The task: given the argument list of a Python function, write out which arguments are required, which are optional and which are `*args`/`**kwargs`.

```
sig: self, name, failobj=None, *rest
out: req: self, name | opt: failobj | var: rest
```

The answer is checkable by a real parser, so every number below is exact, not judged by a model.

**Read `REPORT.md` for the results.** This file explains what is in the repo and how to run it.

## What is here

| Path | What it does |
|---|---|
| `modellab/data.py` | Builds the task from the Python standard library. Train, validation and test are split **by module**, so a test signature never appears in training. |
| `modellab/pretrain.py` | Trains the 5.9M parameter Llama-style base model from scratch on Python source (see "Why a self-trained base model"). |
| `modellab/finetune.py` | One experiment run: full, LoRA or QLoRA training, evaluation, memory and speed measurement. Each run is its own process. |
| `modellab/lora_ref.py` | LoRA written from scratch (under 100 lines). Tests prove it matches the PEFT library. |
| `modellab/nf4_ref.py` | The 4-bit NormalFloat (NF4) quantizer written from scratch. Tests prove it matches bitsandbytes. |
| `modellab/evaluate.py` | Greedy decoding, exact match, field-level F1, forgetting measurement. |
| `modellab/experiments.py` | The experiment plan. Resumable. |
| `modellab/analysis.py` | Aggregates runs into `results/results.json` with confidence intervals. |
| `modellab/bench.py` | Serving speed of full, LoRA (separate and merged) and QLoRA, measured together. |
| `modellab/cli.py` | `predict` with a saved adapter, and `merge`. |
| `scripts/make_report.py` | Writes the tables in `REPORT.md` from `results/results.json`. |
| `models/base/` | The base model and tokenizer used for every reported run. |
| `results/` | Every run's raw JSON, the tuning table, `results.json`. |
| `tests/` | Unit tests, including reference-vs-library equivalence tests. |

## Run it

```bash
pip install -r requirements-dev.txt
python -m pytest -q                       # 48 tests, about 2 minutes on 2 CPU cores

python -m modellab.cli predict --adapter results/adapters/main-mlora-s0 "self, name, failobj=None, *rest"
python -m modellab.cli predict --adapter results/adapters/main-mqlora-s0 --qlora "self, name, failobj=None, *rest"
```

Real output of the first command, which shows the models are not perfect (the right answer has `var: rest`, without the star):

```
self, name, failobj=None, *rest
  -> req: self, name | opt: failobj | var: *rest
```

Reproduce the experiments (about 3 hours on 2 CPU cores, no GPU needed):

```bash
python scripts/build_data.py              # only if you want to rebuild the data
python -m modellab.pretrain --minutes 35  # only if you want to retrain the base model
python -m modellab.experiments all        # baselines, learning rate tuning, main comparison, ablations
python -m modellab.bench                  # serving speed, run on an otherwise idle machine
python -m modellab.analysis               # results/results.json
python scripts/make_report.py             # REPORT.md
```

Finished runs are skipped, so an interrupted `experiments` command resumes where it stopped.

## Method in brief

* **Base model.** 5.87M parameter Llama architecture (6 layers, width 256, 8 heads), 4096 token BPE vocabulary, pretrained on Python source for about 40 minutes of CPU time.
* **Fine-tuning.** Loss only on the answer tokens. AdamW, 6% warmup then cosine decay, batch 16, 4 epochs. LoRA and QLoRA use rank 8, alpha 16, dropout 0.05, on all attention and MLP projections. QLoRA stores the frozen base in 4-bit NF4 with double quantization and computes in bfloat16.
* **Learning rates** are tuned per method on the validation modules only. The test modules are used once per final run, for reporting.
* **Seeds.** 3 seeds for the main comparison, 2 for ablations.
* **Metrics.** Exact match on the whole answer, field-level F1, valid-format rate, trainable parameters, optimizer memory, weight memory, peak process memory, training time, inference speed (adapter unmerged and merged), and forgetting (change in next-token loss on held-out Python source).
* **Uncertainty.** A bootstrap over test questions for the test-set noise, and the spread across seeds for the training noise. Differences between methods use a paired bootstrap on the same questions.

## Why a self-trained base model

The build environment had no access to model hubs, so no pretrained model could be downloaded. The base model is therefore small and trained here. The consequences are real and are discussed in the report: a model this small cannot learn from examples placed in the prompt, so the few-shot baselines score zero, and the full "LoRA vs prompting" question from the roadmap cannot be answered at this scale. The code runs unchanged on a Hugging Face model: set `base` in the run spec to a model directory that has a tokenizer with the same special tokens, or adapt `load_base`.

## Limits

* One task, one small base model, CPU only. Do not read the numbers as a ranking of methods for large models.
* QLoRA here runs bitsandbytes' CPU path. Its speed says little about GPU QLoRA. Its memory and accuracy numbers are real for this model.
* The CI workflow in `.github/workflows` has not been run.
* `data/pretrain/train.txt` (the 48 MB pretraining text) is not shipped. `python scripts/build_data.py` rebuilds it from your own Python install, so counts may differ slightly on another Python version. The trained base model, the task data, the saved adapters and every run are included. Without that file one test (`tests/test_data_and_eval.py`, the pretraining leak check) is skipped, so a fresh unzip reports 47 passed, 1 skipped.
* 4-bit training on CPUs with AVX512-BF16 had a silent-gradient pitfall, covered in `REPORT.md` and guarded in the code.
