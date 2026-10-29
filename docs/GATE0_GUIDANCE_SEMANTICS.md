# Gate 0: VLM guidance semantics audit

Gate 0 must pass before any All-layers G/MG or seed 43/44 training is authorized.
It evaluates one frozen validation batch only. It does not train, select a
checkpoint, or access the locked final test.

## Checks

- G0-1: preprocessing preserves the sample/batch axis.
- G0-2: sample A gives a numerically equivalent eval output alone and in a
  batch. The absolute tolerance is `2e-4` across the standard, duplicated-A,
  and zero-guidance comparisons because changing CUDA batch shape may select a
  different GEMM/SDPA reduction path.
- G0-3: changing only sample B guidance does not change sample A output.
  This same-batch isolation check retains the stricter `1e-6` tolerance and
  also covers preprocessing and the constructed guidance bank.
- G0-4: every predictor block has an explicit, auditable source-layer mapping.
- G0-5: `last` and `all` selectors are behaviorally distinct.
- G0-6: zeroed and shuffled guidance differ from real guidance.

The runtime probe intentionally exits with status 2 when the semantic contract
fails. This is the expected outcome for a detected implementation problem and
must not be bypassed with a skip flag.

## Server command

Run on one free GPU in the existing training environment:

```bash
python3 scripts/gate0_guidance_probe.py \
  --checkpoint outputs/formal_B_alllayers_ddp4_100ep_seed42/ckpt_best.pt \
  --data-dir /path/to/data \
  --cache-dir /path/to/data/cache \
  --gpu 3 \
  --batch-size 2 \
  --output outputs/gate0_B_alllayers_seed42/guidance_semantics.json
```

Preserve the JSON even when the command exits 2. A failed check blocks further
formal training until the implementation is corrected and the probe passes.

## Repair policy after the first failed probe

The first runtime probe on commit `6e26fbf4e67557bf5440ce130f81c0607edbd44c`
confirmed `[B,L,S,D] -> [L,S,D]` batch collapse and no explicit 8-to-12
mapping. The repair keeps the batch axis throughout preprocessing and predictor
conditioning.

For source/target depth mismatch, predictor block `i` receives the nearest
depth-aligned source layer:

```text
source_index(i) = round(i * (source_layers - 1) / (predictor_blocks - 1))
```

For 8 cached layers and 12 predictor blocks this is:

```text
[0, 1, 1, 2, 3, 3, 4, 4, 5, 6, 6, 7]
```

This is a deterministic implementation policy, not a claim that the paper
specified this exact mapping. It must be reported as such and ablated before
formal publication claims.

If the repaired end-to-end probe still fails sample isolation, the v2 report
must decompose the discrepancy into preprocessing, guidance-bank, zero-guidance,
duplicate-A batch, and final predictor-output deltas. Formal training remains
blocked until the first non-isolated stage is identified and corrected.

The v2 runtime evidence separated two defects. Preserving `[B,L,S,D]` and the
explicit layer map made the preprocessed and guidance-bank other-sample deltas
exactly zero, but eval outputs still differed by units. The remaining cause was
SDPA dropout: PyTorch's functional SDPA applies any non-zero `dropout_p` even
while the surrounding module is in eval mode. Commit `bab5a83` now passes zero
dropout to SDPA during eval while preserving the existing training behavior.
After that repair, changing sample B produced an exact-zero delta; batch=1
versus batch=2 left only CUDA reduction roundoff (`4.96e-5`, and `1.12e-4` for
the zero-guidance control), several orders of magnitude below the original
functional failures and the guidance-sensitivity controls.
