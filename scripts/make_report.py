"""Write REPORT.md from results/results.json.

    python -m modellab.analysis && python scripts/make_report.py

Every number in the report comes from results/results.json, the run files under results/runs, or
results/inference_bench.json. The commentary around the numbers was written against the recorded results.
If you rerun the experiments and the numbers move, reread the commentary before publishing it.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = json.loads((ROOT / "results" / "results.json").read_text())
BASE_META = json.loads((ROOT / "models" / "base" / "pretrain_meta.json").read_text())
DATA = json.loads((ROOT / "data" / "summary.json").read_text())

NAMES = {"full": "Full fine-tuning", "lora": "LoRA (r=8)", "qlora": "QLoRA (r=8, 4-bit base)"}


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def ci(row):
    lo, hi = row["em_ci_questions"]
    return f"{pct(lo)} to {pct(hi)}"


def mb(b):
    return f"{b / 1e6:.2f} MB"


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def signed(x: float) -> str:
    v = round(100 * x)
    return "0%" if v == 0 else f"{v:+d}%"


def excludes_zero(lo, hi):
    return lo > 0 or hi < 0


# ---------------------------------------------------------------- sections
def baselines_section():
    b = R["baselines"]
    h = R["heuristics"]
    labels = {"zeroshot": "Zero-shot (instruction only)", "1shot-random": "1 random example in the prompt",
              "3shot-random": "3 random examples in the prompt", "3shot-retrieved": "3 most similar training examples in the prompt"}
    rows = [(labels[k], pct(v["test_em"]), f"{v['test_field_f1']:.3f}", pct(v["valid_rate"])) for k, v in b.items() if k in labels]
    rows.append(("Naive rule: split on commas", pct(h["naive_comma_split_em"]), "n/a", "n/a"))
    rows.append(("Python's own parser (`ast`)", pct(h["reference_parser_em"]), "n/a", "n/a"))
    return table(["Method", "Test exact match", "Field F1", "Valid format"], rows)


def tuning_section():
    t = R["tuning"]
    rows = []
    for m, grid in t["grid"].items():
        for g in grid:
            star = " (chosen)" if abs(t["chosen_lr"][m] - g["lr"]) < 1e-12 else ""
            rows.append((NAMES[m], f"{g['lr']:g}{star}", pct(g["val_em"]), f"{g['final_train_loss']:.3f}"))
    return table(["Method", "Learning rate", "Validation exact match", "Final training loss"], rows)


def main_section():
    rows = []
    for m in ("full", "lora", "qlora"):
        r = R["main"].get(m)
        if not r:
            continue
        sd = f"{pct(r['em_seed_std'])}" if r["em_seed_std"] is not None else "n/a"
        rows.append((NAMES[m], pct(r["em_mean"]), ci(r), sd, f"{r['field_f1']:.3f}", pct(r["valid_rate"])))
    t1 = table(["Method", "Test exact match (mean of seeds)", "95% interval (test questions)", "Std across seeds", "Field F1", "Valid format"], rows)
    rows = []
    for m in ("full", "lora", "qlora"):
        r = R["main"].get(m)
        if not r:
            continue
        rows.append((NAMES[m], f"{r['trainable_params']:,}", pct(r["trainable_fraction"], 2), mb(r["optimizer_state_bytes"]),
                     mb(r["weight_bytes"]), f"{r['peak_rss_mb']:.0f} MB", f"{r['train_seconds']:.0f} s", f"{r['tokens_per_second']:.0f}"))
    t2 = table(["Method", "Trainable parameters", "Share of model", "Optimizer state", "Weights in memory", "Peak process memory", "Training time", "Training tokens/s"], rows)
    rows = []
    for m in ("full", "lora", "qlora"):
        r = R["main"].get(m)
        if not r:
            continue
        rows.append((NAMES[m], f"{r['general_loss_before']:.3f}", f"{r['general_loss_after']:.3f}", f"{r['forgetting_nats']:+.3f}"))
    t3 = table(["Method", "General loss before", "General loss after", "Change (nats)"], rows)
    return t1, t2, t3


def paired_section():
    rows = []
    for label, p in R["paired"].items():
        lo, hi = p["ci_questions"]
        verdict = "interval excludes 0" if excludes_zero(lo, hi) else "interval includes 0"
        rows.append((label, f"{100 * p['diff_em']:+.1f} points", f"{100 * lo:+.1f} to {100 * hi:+.1f}", ", ".join(f"{100 * d:+.1f}" for d in p["diff_by_seed"]), verdict))
    return table(["Comparison (EM difference)", "Mean", "95% interval (paired, test questions)", "By seed", "Reading"], rows)


def inference_section():
    B = R["inference"]
    order = ["full", "lora_separate", "lora_merged", "qlora"]
    names = {"full": "Full model", "lora_separate": "LoRA, adapter kept separate", "lora_merged": "LoRA, adapter merged into the weights",
             "qlora": "QLoRA, 4-bit base + adapter"}
    base1 = B["forms"]["full"]["batch1"]["tokens_per_second_median"]
    base16 = B["forms"]["full"]["batch16"]["tokens_per_second_median"]
    rows = []
    for k in order:
        f = B["forms"][k]
        t1, t16 = f["batch1"]["tokens_per_second_median"], f["batch16"]["tokens_per_second_median"]
        rows.append((names[k], f"{t1:.0f} ({f['batch1']['tokens_per_second_min']:.0f} to {f['batch1']['tokens_per_second_max']:.0f})",
                     signed(t1 / base1 - 1), f"{t16:.0f} ({f['batch16']['tokens_per_second_min']:.0f} to {f['batch16']['tokens_per_second_max']:.0f})",
                     signed(t16 / base16 - 1)))
    return table(["Serving form", "Tokens/s, batch 1 (min to max)", "vs full model", "Tokens/s, batch 16 (min to max)", "vs full model"], rows)


def breakdown_section():
    keys = ["1-2 params", "3-4 params", "5+ params", "simple defaults", "complex defaults"]
    rows = []
    for m in ("full", "lora", "qlora"):
        r = R["main"].get(m)
        if not r:
            continue
        rows.append([NAMES[m]] + [pct(r["breakdown"][k]["em"]) if k in r["breakdown"] else "n/a" for k in keys])
    n = next(iter(R["main"].values()))["breakdown"]
    head = ["Method"] + [f"{k} (n={n[k]['n']})" if k in n else k for k in keys]
    return table(head, rows)


def rank_section():
    rows = []
    for r, v in R["rank"].items():
        rows.append((r, f"{v['trainable_params']:,}", pct(v["em_mean"]), ci(v), f"{v['forgetting_nats']:+.2f}", f"{v['train_seconds']:.0f} s"))
    return table(["Rank", "Trainable parameters", "Test exact match", "95% interval", "Forgetting (nats)", "Training time"], rows)


def targets_section():
    order = ["attn", "mlp", "all"]
    names = {"attn": "Attention only (q, k, v, o)", "mlp": "MLP only (gate, up, down)", "all": "Attention and MLP"}
    rows = []
    for k in order:
        v = R["targets"].get(k)
        if v:
            rows.append((names[k], f"{v['trainable_params']:,}", pct(v["em_mean"]), ci(v), f"{v['forgetting_nats']:+.2f}"))
    return table(["Layers adapted (LoRA r=8)", "Trainable parameters", "Test exact match", "95% interval", "Forgetting (nats)"], rows)


def data_section():
    sizes = sorted({int(n) for m in R["data_size"].values() for n in m})
    rows = []
    for n in sizes:
        row = [str(n)]
        for m in ("full", "lora"):
            v = R["data_size"].get(m, {}).get(str(n))
            row.append(f"{pct(v['em_mean'])} ({ci(v)})" if v else "n/a")
        rows.append(row)
    return table(["Training examples", "Full fine-tuning", "LoRA (r=8)"], rows)


# ---------------------------------------------------------------- numbers that come straight from run files
def per_seed(prefix: str, method: str, key) -> list[float]:
    vals = []
    for p in sorted((ROOT / "results" / "runs").glob(f"{prefix}-m{method}-s*.json")):
        r = json.loads(p.read_text())
        vals.append(key(r))
    return vals


def spread(vals: list[float], fmt="{:.3f}") -> str:
    return f"{fmt.format(sum(vals) / len(vals))} (range {fmt.format(min(vals))} to {fmt.format(max(vals))})"


def gap_table() -> str:
    rows = []
    for m in ("full", "lora", "qlora"):
        val = per_seed("main", m, lambda r: r["val"]["em"])
        test = per_seed("main", m, lambda r: r["test"]["em"])
        tl = per_seed("main", m, lambda r: r["training"]["final_train_loss"])
        forget = per_seed("main", m, lambda r: r["forgetting_nats"])
        rows.append((NAMES[m], pct(sum(val) / len(val)), pct(sum(test) / len(test)),
                     f"{100 * (sum(val) / len(val) - sum(test) / len(test)):+.1f} points", f"{sum(tl) / len(tl):.3f}", spread(forget, "{:.2f}")))
    return table(["Method", "Validation exact match", "Test exact match", "Validation minus test", "Final training loss", "Forgetting in nats (mean, range over seeds)"], rows)


# ---------------------------------------------------------------- the document
def build() -> str:
    m = R["main"]
    f, l, q = m["full"], m["lora"], m["qlora"]
    P = R["paired"]
    lf, ql, qf = P["lora vs full"], P["qlora vs lora"], P["qlora vs full"]
    rk, tg, ds = R["rank"], R["targets"], R["data_size"]
    t1, t2, t3 = main_section()
    B = R["inference"]["forms"]
    forget = {k: per_seed("main", k, lambda r: r["forgetting_nats"]) for k in ("full", "lora", "qlora")}
    opt_ratio = f["optimizer_state_bytes"] / l["optimizer_state_bytes"]
    w_ratio = l["weight_bytes"] / q["weight_bytes"]
    sep_slow = 100 * (1 - B["lora_separate"]["batch1"]["tokens_per_second_median"] / B["full"]["batch1"]["tokens_per_second_median"])
    qlo_slow = 100 * (1 - B["qlora"]["batch1"]["tokens_per_second_median"] / B["full"]["batch1"]["tokens_per_second_median"])
    ds_gaps = {n: ds["lora"][n]["em_mean"] - ds["full"][n]["em_mean"] for n in ds["lora"] if n in ds["full"]}
    task = DATA["task"]
    min_rss = min(float(json.loads(p.read_text())["peak_rss_mb"]) for p in (ROOT / "results" / "runs").glob("data-*.json"))

    return f"""# ModelLab: LoRA, QLoRA and full fine-tuning, measured

One small language model, one narrow task, three ways of adapting it, and a record of what each one costs. Everything here was run on 2 CPU cores with no GPU. All numbers are on test modules that were never seen in training.

## The short version

1. **LoRA beat full fine-tuning here.** Exact match on the test set: LoRA {pct(l['em_mean'])}, QLoRA {pct(q['em_mean'])}, full fine-tuning {pct(f['em_mean'])} (mean of 3 seeds). LoRA minus full is {100 * lf['diff_em']:+.1f} points, 95% interval {100 * lf['ci_questions'][0]:+.1f} to {100 * lf['ci_questions'][1]:+.1f}, and the sign is the same in every seed. LoRA also stayed ahead at every training-set size I tried (100, 300, 1000 and 2036 examples).
2. **QLoRA matched LoRA.** The difference is {100 * ql['diff_em']:+.1f} points (interval {100 * ql['ci_questions'][0]:+.1f} to {100 * ql['ci_questions'][1]:+.1f}), which is noise. QLoRA keeps the frozen model in {w_ratio:.1f}x less memory ({mb(q['weight_bytes'])} against {mb(l['weight_bytes'])}).
3. **Few-shot prompting scored 0% in all four setups.** The base model is far too small to learn from examples placed in a prompt. So the roadmap question "does LoRA beat few-shot prompting" is **not** answered by this project. A 0% baseline says nothing about larger models.
4. **This task is easy for a rule.** Splitting on commas gets {pct(R['heuristics']['naive_comma_split_em'])} and Python's own parser gets 100%. The fine-tuned models reach about 79%. This project compares adaptation methods. It does not build a useful parser.
5. **LoRA's savings were smaller than usually claimed, on a model this small.** It trains {pct(l['trainable_fraction'])} of the weights and its optimizer state is {opt_ratio:.0f}x smaller ({mb(l['optimizer_state_bytes'])} against {mb(f['optimizer_state_bytes'])}). But training was slower ({l['train_seconds']:.0f} s against {f['train_seconds']:.0f} s) and peak process memory was not lower ({l['peak_rss_mb']:.0f} MB against {f['peak_rss_mb']:.0f} MB).
6. **Nothing here shows LoRA forgetting less.** All three methods made the model worse at general Python by about 1 nat (loss on general Python rose from 2.83 to about 3.8). Higher LoRA rank forgot more with no accuracy gain.
7. **A silent bug in 4-bit training on some CPUs** nearly produced a false "QLoRA does not work" result. It is described below, with the test that now guards against it.

## What was built

**Task.** Given the argument list of a Python function, write which arguments are required, optional, or `*args`/`**kwargs`.

```
sig: self, name, failobj=None, *rest
out: req: self, name | opt: failobj | var: rest
```

A real parser can check the answer, so scores are exact. The data comes from the Python 3.13 standard library: {task['train']['examples']} training, {task['val']['examples']} validation and {task['test']['examples']} test signatures. The split is **by module** ({task['train']['modules']}, {task['val']['modules']} and {task['test']['modules']} modules), so a test signature never appears in training, and I checked that no test or validation signature appears in the pretraining text either.

**Base model.** A 5.87M parameter Llama-style model (6 layers, width 256) trained from scratch on Python source: {BASE_META['tokens_seen'] / 1e6:.1f}M tokens, about {BASE_META['wall_seconds'] / 60:.0f} minutes, final loss on held-out Python {BASE_META['final_valid_loss_nats']:.2f} nats (perplexity {BASE_META['final_valid_perplexity']:.1f}). I trained it because the build environment could not reach any model hub. A pretrained model of a useful size would change the numbers, and probably the few-shot result.

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

{baselines_section()}

The prompted models never produced the answer format, so field F1 is 0 as well. Their outputs are repeating fragments of Python. The two rule-based rows are references, not competitors.

### Learning rate search (validation modules only)

{tuning_section()}

Every method's best value sits inside its grid, not on an edge. (Full fine-tuning's best was first on the edge, so I extended its grid downward and reran.)

### Main comparison (3 seeds each)

{t1}

{paired_section()}

### What each method costs

{t2}

* Weights in memory: for LoRA this is the frozen base plus the adapter. For QLoRA the linear layers are 4-bit but the embedding table and norms stay in full precision, which is why the saving is {w_ratio:.1f}x and not 8x.
* Peak process memory is the whole Python process. The smallest runs in this project peaked at {min_rss:.0f} MB, so most of the figure is the PyTorch runtime and not the model. Read the optimizer-state column for the effect of LoRA on training memory.
* Training time is slower for LoRA and QLoRA than for full fine-tuning. The backward pass still runs through the whole network and LoRA adds extra small matrix products to every adapted layer, which is likely a large fraction of the work on a model this small. I did not profile it. QLoRA is slowest, probably because it also unpacks 4-bit weights on the CPU.

### Serving speed (measured together, 5 interleaved rounds on an idle machine)

{inference_section()}

A merged adapter serves at the speed of a plain model, as it should: after merging there is no adapter left. Keeping the adapter separate costs about {sep_slow:.0f}% of the speed at batch size 1. QLoRA costs about {qlo_slow:.0f}%, on this CPU path. Do not carry the QLoRA number over to a GPU, where 4-bit kernels behave differently. The min to max column shows how noisy a shared 2-core machine is: treat differences under about 10% as unresolved.

### Forgetting

{t3}

"General loss" is next-token loss on held-out Python source that has nothing to do with the task. Lower is better. All three methods end about 0.9 to 1.0 nats higher: full {spread(forget['full'], '{:.2f}')}, LoRA {spread(forget['lora'], '{:.2f}')}, QLoRA {spread(forget['qlora'], '{:.2f}')}. If anything LoRA forgot slightly more than full fine-tuning, but that gap (about 0.1) is no bigger than the spread between seeds of a single method, so I would not claim either way. This is a large drop for every method: these runs push a small model hard onto one narrow output format.

### Where the models fail

{breakdown_section()}

Accuracy falls steeply with the number of arguments: the models are good on short signatures and poor on long ones. Signatures whose defaults contain brackets, quotes or calls are also harder. A comma-splitting rule is nearly perfect, so the failures are about what a 6M parameter model can copy exactly, not about task difficulty.

### Why was LoRA ahead of full fine-tuning?

{gap_table()}

Full fine-tuning drives the training loss far lower ({sum(per_seed('main', 'full', lambda r: r['training']['final_train_loss'])) / 3:.3f} against {sum(per_seed('main', 'lora', lambda r: r['training']['final_train_loss'])) / 3:.3f}) and scores about 9 points lower on test modules than on validation modules. LoRA shows no such gap. So the pattern fits full fine-tuning fitting the training modules more tightly and transferring less well to new ones. That is a reading of these numbers, not something I tested. Things that could narrow the gap and that I did not try: early stopping, weight decay, more epochs at a lower rate, a larger training set (the gap did not close at 2036 examples, see below).

## Ablations (2 seeds each, LoRA at the learning rate chosen for rank 8)

The learning rate was **not** retuned per setting. A different rank or layer choice might prefer a different rate, so read these as "with everything else held fixed".

**Rank**

{rank_section()}

Accuracy rises quickly up to rank 8 and then stops: ranks 8, 16 and 32 are within each other's intervals. Forgetting goes the other way and keeps rising ({rk['1']['forgetting_nats']:+.2f} at rank 1, {rk['8']['forgetting_nats']:+.2f} at rank 8, {rk['32']['forgetting_nats']:+.2f} at rank 32), and training time stays between about 200 and 230 s for every rank (rank 8 comes from the main runs, made at a different time, so differences of that size are not meaningful). In this setup rank 8 is the cheapest point that reaches the plateau. Part of the forgetting rise may come from the fixed learning rate acting on a bigger update at higher rank.

**Which layers get an adapter**

{targets_section()}

Adapting both attention and MLP layers was best. Attention-only ({pct(tg['attn']['em_mean'])}) beat MLP-only ({pct(tg['mlp']['em_mean'])}) with fewer trainable parameters ({tg['attn']['trainable_params']:,} against {tg['mlp']['trainable_params']:,}), and forgot less than adapting both ({tg['attn']['forgetting_nats']:+.2f} against {tg['all']['forgetting_nats']:+.2f} nats). The task is mostly about copying argument names from the input, which is plausibly an attention job. I did not test that.

**Training-set size** (full fine-tuning at its chosen rate, LoRA at its chosen rate; the number of steps has a floor of 150, so the smallest sets get many passes)

{data_section()}

The 2036 row uses seeds 0 and 1 only, so it differs a little from the 3-seed main table. LoRA is ahead at every size: by {100 * ds_gaps['100']:.1f}, {100 * ds_gaps['300']:.1f}, {100 * ds_gaps['1000']:.1f} and {100 * ds_gaps['2036']:.1f} points. The test-question intervals do not overlap at any size. LoRA with 1000 examples ({pct(ds['lora']['1000']['em_mean'])}) matches full fine-tuning with all 2036 ({pct(ds['full']['2036']['em_mean'])}). Both methods are still improving at 2036 examples.

## A pitfall: silent wrong gradients in 4-bit training

My first QLoRA run scored 0% on the validation modules while the same setup with LoRA scored 65%, and the training loss for QLoRA stalled at 1.39 while LoRA's reached 0.15. There was no error and no warning.

Cause: on CPUs with AVX512-BF16 support, bitsandbytes converts each 4-bit layer to an inference-only packed format the first time that layer runs in eval mode on an input without gradients. That format has no backward pass. My run script measured the base model's loss before training, which was exactly such a forward. After it, gradients no longer flowed through the frozen layers, so only the adapters closest to the output got a correct gradient. I confirmed this by comparing adapter gradients against a full-precision model holding the same dequantized weights: mean cosine similarity across the 84 adapter tensors was 1.000 before the inference-only forward and about 0 (minimum -0.16) after it. (Single layers tested in isolation looked fine, which is why this took a while to find.)

Fix: measure the quantized base on a throwaway copy, and refuse to train if any layer has already been switched (`assert_backward_ready` in `modellab/finetune.py`). Three tests in `tests/test_finetune.py` cover it, and I checked they fail on the broken behaviour. The invalid run is kept in `results/discarded/` and is not in any table. Whether this bites depends on the CPU, so a QLoRA setup that works on one machine can silently underperform on another.

## What this does and does not show

* **One task, one tiny base model, CPU only.** This is not a ranking of methods for large models. In particular, "LoRA beats full fine-tuning" is a result for a 6M parameter model trained on about 2000 examples. With more data or a larger model the order may change.
* **The few-shot baselines are uninformative.** A model this small cannot do in-context learning, so 0% is expected and says nothing about prompting a capable model.
* **Learning rates were tuned on one seed, with 3 to 5 values per method.** A finer search could move any method by a few points. Full fine-tuning got the widest grid. The number of epochs (4) was fixed for all methods, not tuned.
* **Test set size.** {task['test']['examples']} questions from {task['test']['modules']} modules. The intervals above reflect question-level noise. They do not reflect the choice of which modules ended up in the test set.
* **Per-run speed numbers recorded during training were wrong** (a bfloat16 autocast caching effect favoured the full model) and were replaced by the separate benchmark above. The old numbers are still inside three run files under `inference` and are not used.
* **The QLoRA speed and memory figures are for bitsandbytes on CPU.** The accuracy comparison carries over more easily than the speed one.
* **Forgetting** is measured only as loss on held-out Python source. I did not test other abilities.

## Reproduce

See `README.md`. In short: `python -m modellab.experiments all`, then `python -m modellab.bench`, then `python -m modellab.analysis && python scripts/make_report.py`. The 50 runs behind the tables are in `results/runs/`.
"""


if __name__ == "__main__":
    out = ROOT / "REPORT.md"
    out.write_text(build())
    print("wrote", out, f"({len(out.read_text().splitlines())} lines)")
