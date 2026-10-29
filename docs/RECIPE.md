# Maya recipe on 2× RTX 2080 Ti 22 GB (NVLink)

Hardware: Xeon E5-2686 v4 (AVX2), quad-channel DDR3-1600, 247 GB RAM, 2× RTX 2080 Ti modded to 22 GB (sm_75, no bf16), NVLink NV2 (2 links × 25.8 GB/s per direction ≈ 51.6 GB/s each way), PCIe 3.0 x16. Model Maya-S24 on a tmpfs (`/dev/shm`, 91 GB). Maya v1.1.36.

## Best recipe so far (Maya-S24, 256K × 4 users)

- Quarantine holder running first (see README).
- **Production `build/maya`**, not a debug/trace build.
- Env: `MAYA_GLM_DUAL_PREFILL=1` (uses NVLink for expert hand-offs). Nothing else.
- Config: `config/maya-s24-prod.json` (`--max-context 262144 --parallel 4 --kv int8 --expert-cache auto --prefill auto --kv-streaming on`).
- Do **not** set `STRATA_GLM_VRAM_GB=0,0` on the production build: it is read as a literal 0 GB cap and the engine exits with "no room for the expert pool". The holder already protects the bad band.
- GPU clocks are left at the driver's dynamic, power-capped behaviour (not pinned).

## Results (fresh random prompts with a nonce at the start, so no prefix-cache hits)

| Test | Result |
|---|---|
| Prefill 3k tokens | ~5.6 s (~540 tok/s) |
| Prefill 8k tokens | ~12.8 s (~625 tok/s), twice |
| Same prefill on the debug/trace build | ~35 tok/s (about 17× slower) |
| Decode, single stream | 27–31 tok/s |
| 4 concurrent users, 500 tokens each | ~27 tok/s aggregate (production build) |
| Router / non-finite errors with holder | 0 |

## Tuning that did not help (one change at a time)

| Change | Effect |
|---|---|
| `MAYA_GLM_PREFILL_MIN_CHUNKS=4` | no gain (3k got worse, 8k ~600 tok/s) |
| `MAYA_GLM_PREFILL_ATTN=wmma` | no change |
| `MAYA_GLM_PREFILL_CALC` | only for GPUs without NVLink |

Prefill plateaus at ~620 tok/s and is compute-bound per GPU (CUDA0 ~6.6 s, CUDA1 ~6.0 s on 8k).

## Pitfalls

- Early "0.1–0.2 s" prefill readings were prefix-cache hits. Put a random nonce at the start of every benchmark prompt.
- Loading takes 2–5 minutes.
- Stop Maya completely before restarting and wait for VRAM to drop (the holder keeps ~1.2 GB on GPU 0).

## Pending

Maya-M with a 1M-token KV cache and its benchmark are not done yet; this file is updated when they are.
