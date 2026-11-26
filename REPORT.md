# ModelLab: LoRA, QLoRA and full fine-tuning, measured

One small language model, one narrow task, three ways of adapting it, and a record of what each one costs. Everything here was run on 2 CPU cores with no GPU. All numbers are on test modules that were never seen in training.

## The short version

1. **LoRA beat full fine-tuning here.** Exact match on the test set: LoRA 78.8%, QLoRA 78.2%, full fine-tuning 66.1% (mean of 3 seeds). LoRA minus full is +12.7 points, 95% interval +9.9 to +15.5, and the sign is the same in every seed. LoRA also stayed ahead at every training-set size I tried (100, 300, 1000 and 2036 examples).
2. **QLoRA matched LoRA.** The difference is -0.6 points (interval -1.7 to +0.6), which is noise. QLoRA keeps the frozen model in 3.2x less memory (7.56 MB against 24.42 MB).
3. **Few-shot prompting scored 0% in all four setups.** The base model is far too small to learn from examples placed in a prompt. So the roadmap question "does LoRA beat few-shot prompting" is **not** answered by this project. A 0% baseline says nothing about larger models.
4. **This task is easy for a rule.** Splitting on commas gets 99.7% and Python's own parser gets 100%. The fine-tuned models reach about 79%. This project compares adaptation methods. It does not build a useful parser.
5. **LoRA's savings were smaller than usually claimed, on a model this small.** It trains 3.9% of the weights and its optimizer state is 25x smaller (1.89 MB against 46.95 MB). But training was slower (225 s against 137 s) and peak process memory was not lower (2700 MB against 2338 MB).
6. **Nothing here shows LoRA forgetting less.** All three methods made the model worse at general Python by about 1 nat (loss on general Python rose from 2.83 to about 3.8). Higher LoRA rank forgot more with no accuracy gain.
7. **A silent bug in 4-bit training on some CPUs** nearly produced a false "QLoRA does not work" result. It is described below, with the test that now guards against it.

## What was built

**Task.** Given the argument list of a Python function, write which arguments are required, optional, or `*args`/`**kwargs`.

```
sig: self, name, failobj=None, *rest
out: req: self, name | opt: failobj | var: rest
```

A real parser can check the answer, so scores are exact. The data comes from the Python 3.13 standard library: 2036 training, 324 validation and 798 test signatures. The split is **by module** (114, 8 and 38 modules), so a test signature never appears in training, and I checked that no test or validation signature appears in the pretraining text either.

**Base model.** A 5.87M parameter Llama-style model (6 layers, width 256) trained from scratch on Python source: 11.5M tokens, about 40 minutes, final loss on held-out Python 2.79 nats (perplexity 16.3). I trained it because the build environment could not reach any model hub. A pretrained model of a useful size would change the numbers, and probably the few-shot result.

**Methods.** Same data, same loss (answer tokens only), same 4 epochs (512 steps, batch 16), same optimizer (AdamW, warmup then cosine decay), different learning rate per method.

| Method | What is trained |
|---|---|
| Full fine-tuning | every weight |
| LoRA | small low-rank matrices added to every attention and MLP projection (rank 8, alpha 16, dropout 0.05). The base is frozen |
| QLoRA | the same LoRA, on a base stored in 4-bit NF4 with double quantization, computing in bfloat16 |

**How the learning rate was chosen.** Each method got a grid of learning rates, one seed, scored on the validation modules only. The best one was then used for 3 seeds scored on the test modules. The test modules were not used to choose anything.

**Uncertainty.** Two sources of noise are reported separately. The 95% intervals come from resampling the test questions (test-set noise). The "std across seeds" is training noise. Differences between methods use a paired resampling of the same questions.

## Results

### Baselines that need no training

| Method | Test exact match | Field F1 | Valid format |
|---|---|---|---|
| 1 random example in the prompt | 0.0% | 0.000 | 0.0% |
| 3 random examples in the prompt | 0.0% | 0.000 | 0.0% |
| 3 most similar training examples in the prompt | 0.0% | 0.000 | 0.0% |
| Zero-shot (instruction only) | 0.0% | 0.000 | 0.0% |
| Naive rule: split on commas | 99.7% | n/a | n/a |
| Python's own parser (`ast`) | 100.0% | n/a | n/a |

The prompted models never produced the answer format, so field F1 is 0 as well. Their outputs are repeating fragments of Python. The two rule-based rows are references, not competitors.

### Learning rate search (validation modules only)

| Method | Learning rate | Validation exact match | Final training loss |
|---|---|---|---|
| Full fine-tuning | 3e-05 | 59.3% | 0.169 |
| Full fine-tuning | 0.0001 | 70.4% | 0.047 |
| Full fine-tuning | 0.0003 (chosen) | 76.2% | 0.017 |
| Full fine-tuning | 0.001 | 71.6% | 0.010 |
| Full fine-tuning | 0.003 | 56.8% | 0.051 |
| LoRA (r=8) | 0.001 | 64.5% | 0.154 |
| LoRA (r=8) | 0.003 (chosen) | 78.7% | 0.100 |
| LoRA (r=8) | 0.01 | 60.2% | 0.246 |
| QLoRA (r=8, 4-bit base) | 0.001 | 67.0% | 0.156 |
| QLoRA (r=8, 4-bit base) | 0.003 (chosen) | 76.9% | 0.098 |
| QLoRA (r=8, 4-bit base) | 0.01 | 57.4% | 0.247 |

Every method's best value sits inside its grid, not on an edge. (Full fine-tuning's best was first on the edge, so I extended its grid downward and reran.)

### Main comparison (3 seeds each)

| Method | Test exact match (mean of seeds) | 95% interval (test questions) | Std across seeds | Field F1 | Valid format |
|---|---|---|---|---|---|
| Full fine-tuning | 66.1% | 62.9% to 69.2% | 0.2% | 0.858 | 98.2% |
| LoRA (r=8) | 78.8% | 76.2% to 81.2% | 1.0% | 0.914 | 98.8% |
| QLoRA (r=8, 4-bit base) | 78.2% | 75.6% to 80.7% | 0.8% | 0.912 | 98.6% |

| Comparison (EM difference) | Mean | 95% interval (paired, test questions) | By seed | Reading |
|---|---|---|---|---|
| lora vs full | +12.7 points | +9.9 to +15.5 | +11.9, +12.7, +13.7 | interval excludes 0 |
| qlora vs lora | -0.6 points | -1.7 to +0.6 | +1.1, -0.5, -2.4 | interval includes 0 |
| qlora vs full | +12.2 points | +9.4 to +14.9 | +13.0, +12.2, +11.3 | interval excludes 0 |

### What each method costs

| Method | Trainable parameters | Share of model | Optimizer state | Weights in memory | Peak process memory | Training time | Training tokens/s |
|---|---|---|---|---|---|---|---|
| Full fine-tuning | 5,868,800 | 100.00% | 46.95 MB | 23.48 MB | 2338 MB | 137 s | 2125 |
| LoRA (r=8) | 236,544 | 3.87% | 1.89 MB | 24.42 MB | 2700 MB | 225 s | 1293 |
| QLoRA (r=8, 4-bit base) | 236,544 | 3.87% | 1.89 MB | 7.56 MB | 2799 MB | 251 s | 1160 |

* Weights in memory: for LoRA this is the frozen base plus the adapter. For QLoRA the linear layers are 4-bit but the embedding table and norms stay in full precision, which is why the saving is 3.2x and not 8x.
* Peak process memory is the whole Python process. The smallest runs in this project peaked at 1427 MB, so most of the figure is the PyTorch runtime and not the model. Read the optimizer-state column for the effect of LoRA on training memory.
* Training time is slower for LoRA and QLoRA than for full fine-tuning. The backward pass still runs through the whole network and LoRA adds extra small matrix products to every adapted layer, which is likely a large fraction of the work on a model this small. I did not profile it. QLoRA is slowest, probably because it also unpacks 4-bit weights on the CPU.

### Serving speed (measured together, 5 interleaved rounds on an idle machine)

| Serving form | Tokens/s, batch 1 (min to max) | vs full model | Tokens/s, batch 16 (min to max) | vs full model |
|---|---|---|---|---|
| Full model | 148 (120 to 158) | 0% | 1345 (1156 to 1734) | 0% |
| LoRA, adapter kept separate | 102 (90 to 108) | -31% | 906 (766 to 963) | -33% |
| LoRA, adapter merged into the weights | 147 (126 to 155) | 0% | 1273 (1201 to 1375) | -5% |
| QLoRA, 4-bit base + adapter | 91 (85 to 100) | -39% | 724 (651 to 753) | -46% |

A merged adapter serves at the speed of a plain model, as it should: after merging there is no adapter left. Keeping the adapter separate costs about 31% of the speed at batch size 1. QLoRA costs about 39%, on this CPU path. Do not carry the QLoRA number over to a GPU, where 4-bit kernels behave differently. The min to max column shows how noisy a shared 2-core machine is: treat differences under about 10% as unresolved.

### Forgetting

| Method | General loss before | General loss after | Change (nats) |
|---|---|---|---|
| Full fine-tuning | 2.825 | 3.756 | +0.931 |
| LoRA (r=8) | 2.825 | 3.846 | +1.021 |
| QLoRA (r=8, 4-bit base) | 2.835 | 3.778 | +0.943 |

"General loss" is next-token loss on held-out Python source that has nothing to do with the task. Lower is better. All three methods end about 0.9 to 1.0 nats higher: full 0.93 (range 0.91 to 0.95), LoRA 1.02 (range 0.96 to 1.10), QLoRA 0.94 (range 0.87 to 1.01). If anything LoRA forgot slightly more than full fine-tuning, but that gap (about 0.1) is no bigger than the spread between seeds of a single method, so I would not claim either way. This is a large drop for every method: these runs push a small model hard onto one narrow output format.

### Where the models fail

| Method | 1-2 params (n=315) | 3-4 params (n=343) | 5+ params (n=140) | simple defaults (n=710) | complex defaults (n=88) |
|---|---|---|---|---|---|
| Full fine-tuning | 77.2% | 71.0% | 28.8% | 68.3% | 48.5% |
| LoRA (r=8) | 91.3% | 83.5% | 39.3% | 81.0% | 61.0% |
| QLoRA (r=8, 4-bit base) | 90.4% | 82.2% | 41.2% | 80.5% | 59.9% |

Accuracy falls steeply with the number of arguments: the models are good on short signatures and poor on long ones. Signatures whose defaults contain brackets, quotes or calls are also harder. A comma-splitting rule is nearly perfect, so the failures are about what a 6M parameter model can copy exactly, not about task difficulty.

### Why was LoRA ahead of full fine-tuning?

| Method | Validation exact match | Test exact match | Validation minus test | Final training loss | Forgetting in nats (mean, range over seeds) |
|---|---|---|---|---|---|
| Full fine-tuning | 75.6% | 66.1% | +9.5 points | 0.016 | 0.93 (range 0.91 to 0.95) |
| LoRA (r=8) | 79.1% | 78.8% | +0.3 points | 0.097 | 1.02 (range 0.96 to 1.10) |
| QLoRA (r=8, 4-bit base) | 77.8% | 78.2% | -0.5 points | 0.097 | 0.94 (range 0.87 to 1.01) |

Full fine-tuning drives the training loss far lower (0.016 against 0.097) and scores about 9 points lower on test modules than on validation modules. LoRA shows no such gap. So the pattern fits full fine-tuning fitting the training modules more tightly and transferring less well to new ones. That is a reading of these numbers, not something I tested. Things that could narrow the gap and that I did not try: early stopping, weight decay, more epochs at a lower rate, a larger training set (the gap did not close at 2036 examples, see below).

## Ablations (2 seeds each, LoRA at the learning rate chosen for rank 8)

The learning rate was **not** retuned per setting. A different rank or layer choice might prefer a different rate, so read these as "with everything else held fixed".

**Rank**

| Rank | Trainable parameters | Test exact match | 95% interval | Forgetting (nats) | Training time |
|---|---|---|---|---|---|
| 1 | 29,568 | 57.0% | 53.9% to 60.0% | +0.65 | 200 s |
| 2 | 59,136 | 66.5% | 63.5% to 69.5% | +0.69 | 202 s |
| 4 | 118,272 | 74.3% | 71.4% to 77.1% | +0.91 | 202 s |
| 8 | 236,544 | 78.3% | 75.6% to 80.8% | +1.05 | 226 s |
| 16 | 473,088 | 78.5% | 75.8% to 81.0% | +1.32 | 204 s |
| 32 | 946,176 | 77.2% | 74.6% to 79.6% | +2.19 | 207 s |

Accuracy rises quickly up to rank 8 and then stops: ranks 8, 16 and 32 are within each other's intervals. Forgetting goes the other way and keeps rising (+0.65 at rank 1, +1.05 at rank 8, +2.19 at rank 32), and training time stays between about 200 and 230 s for every rank (rank 8 comes from the main runs, made at a different time, so differences of that size are not meaningful). In this setup rank 8 is the cheapest point that reaches the plateau. Part of the forgetting rise may come from the fixed learning rate acting on a bigger update at higher rank.

**Which layers get an adapter**

| Layers adapted (LoRA r=8) | Trainable parameters | Test exact match | 95% interval | Forgetting (nats) |
|---|---|---|---|---|
| Attention only (q, k, v, o) | 98,304 | 72.1% | 69.2% to 74.8% | +0.67 |
| MLP only (gate, up, down) | 138,240 | 66.8% | 63.7% to 69.8% | +0.94 |
| Attention and MLP | 236,544 | 78.3% | 75.6% to 80.8% | +1.05 |

Adapting both attention and MLP layers was best. Attention-only (72.1%) beat MLP-only (66.8%) with fewer trainable parameters (98,304 against 138,240), and forgot less than adapting both (+0.67 against +1.05 nats). The task is mostly about copying argument names from the input, which is plausibly an attention job. I did not test that.

**Training-set size** (full fine-tuning at its chosen rate, LoRA at its chosen rate; the number of steps has a floor of 150, so the smallest sets get many passes)

| Training examples | Full fine-tuning | LoRA (r=8) |
|---|---|---|
| 100 | 23.9% (21.3% to 26.5%) | 32.8% (29.8% to 35.8%) |
| 300 | 42.9% (39.9% to 45.9%) | 51.6% (48.4% to 54.7%) |
| 1000 | 58.8% (55.7% to 61.8%) | 66.8% (63.8% to 69.8%) |
| 2036 | 66.0% (62.8% to 69.0%) | 78.3% (75.6% to 80.8%) |

The 2036 row uses seeds 0 and 1 only, so it differs a little from the 3-seed main table. LoRA is ahead at every size: by 9.0, 8.8, 8.0 and 12.3 points. The test-question intervals do not overlap at any size. LoRA with 1000 examples (66.8%) matches full fine-tuning with all 2036 (66.0%). Both methods are still improving at 2036 examples.

## A pitfall: silent wrong gradients in 4-bit training

My first QLoRA run scored 0% on the validation modules while the same setup with LoRA scored 65%, and the training loss for QLoRA stalled at 1.39 while LoRA's reached 0.15. There was no error and no warning.

Cause: on CPUs with AVX512-BF16 support, bitsandbytes converts each 4-bit layer to an inference-only packed format the first time that layer runs in eval mode on an input without gradients. That format has no backward pass. My run script measured the base model's loss before training, which was exactly such a forward. After it, gradients no longer flowed through the frozen layers, so only the adapters closest to the output got a correct gradient. I confirmed this by comparing adapter gradients against a full-precision model holding the same dequantized weights: mean cosine similarity across the 84 adapter tensors was 1.000 before the inference-only forward and about 0 (minimum -0.16) after it. (Single layers tested in isolation looked fine, which is why this took a while to find.)

Fix: measure the quantized base on a throwaway copy, and refuse to train if any layer has already been switched (`assert_backward_ready` in `modellab/finetune.py`). Three tests in `tests/test_finetune.py` cover it, and I checked they fail on the broken behaviour. The invalid run is kept in `results/discarded/` and is not in any table. Whether this bites depends on the CPU, so a QLoRA setup that works on one machine can silently underperform on another.

## What this does and does not show

* **One task, one tiny base model, CPU only.** This is not a ranking of methods for large models. In particular, "LoRA beats full fine-tuning" is a result for a 6M parameter model trained on about 2000 examples. With more data or a larger model the order may change.
* **The few-shot baselines are uninformative.** A model this small cannot do in-context learning, so 0% is expected and says nothing about prompting a capable model.
* **Learning rates were tuned on one seed, with 3 to 5 values per method.** A finer search could move any method by a few points. Full fine-tuning got the widest grid. The number of epochs (4) was fixed for all methods, not tuned.
* **Test set size.** 798 questions from 38 modules. The intervals above reflect question-level noise. They do not reflect the choice of which modules ended up in the test set.
* **Per-run speed numbers recorded during training were wrong** (a bfloat16 autocast caching effect favoured the full model) and were replaced by the separate benchmark above. The old numbers are still inside three run files under `inference` and are not used.
* **The QLoRA speed and memory figures are for bitsandbytes on CPU.** The accuracy comparison carries over more easily than the speed one.
* **Forgetting** is measured only as loss on held-out Python source. I did not test other abilities.

## Reproduce

See `README.md`. In short: `python -m modellab.experiments all`, then `python -m modellab.bench`, then `python -m modellab.analysis && python scripts/make_report.py`. The 50 runs behind the tables are in `results/runs/`.
