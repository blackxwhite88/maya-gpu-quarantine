# Can GPU 0's bad VRAM band be blocked off? (2026-10-29)

## Conclusion

**Yes. A helper can hold the bad physical memory, and tests show it works.** A small CUDA tool (`bandmap`,
source in the appendix) splits GPU 0's free VRAM into separate 2 MiB physical allocations (`cuMemCreate`),
pattern-tests every chunk, frees the good ones and keeps holding the bad ones. While it holds them, a **fresh,
unrelated process** that fills the rest of the free memory with test patterns sees **zero** bit flips. That held
for a 10-pass `cuMemCreate` test and for one large plain `cudaMalloc` (the same kind of allocation as Maya's
expert pool). With no holder, the same plain test, run straight afterwards, flips about 9,000 words in the
same place.

Holding works because a held allocation is physical memory the driver cannot hand to anyone else. It does not
depend on a virtual offset staying put, and that matters: virtual offsets do **not** stay put (see Q1).

How much of the card can be made safe (CUDA reports 21,999 MiB usable on GPU 0; the other 529 MiB is reserved
by the driver):

| Method | Held | Helper context | Safe and usable | vs today's 12.5 GB cap |
|---|---|---|---|---|
| **Span hold** (recommended): every chunk from the first bad to the last, plus 16 chunks (32 MiB) each side | 1,030 MiB | ~158 MiB | **~20,800 MiB (94.6 %)** | about +8 GB |
| Cluster hold: the bad chunks found, plus 2 chunks each side | 198 MiB | ~158 MiB | ~21,640 MiB (98.4 %) | about +9 GB, less margin |
| Either of these done inside Maya (the Q3 patch) | same | 0 | about +158 MiB more | |

**Recommendation:** use the span hold. It is cheap (1 GB), and the bad set flickers from run to run (whole
clusters come and go, see the map below), so a hold of only the chunks seen bad can miss a weak cluster that a
later run would have caught. Before Maya's cap comes off, run the whole-card window test (plan below), because
two things can only be checked on an empty card: whether Maya's own lower 12.4 GB is clean, and whether the
band sits at the same physical place after a reboot.

## Q1: can a helper hold the band, and is the placement stable?

- **Granularity.** `cuMemGetAllocationGranularity` on GPU 0 gives minimum = recommended = **2 MiB**. Through
  CUDA, 2 MiB is the smallest unit that can be held. Inside each bad 2 MiB chunk only 1 to 5 of its 32 64 KiB
  pages ever flipped; every page index 0..31 shows up across chunks, with no favoured position. A 64 KiB hold
  would be about 10x tighter, but CUDA does not offer it.
- **The physical allocator works bottom-up and in order.** Chunks allocated in order show bad clusters at a
  regular period: about 6 chunks (12 MiB) every about 43.5 chunks (87 MiB), 11 to 12 clusters across about
  966 MiB. Example: chunk indexes 482-486, 525-530, 568-573, 611-616, 655-660, 698-703, 742-746, 785-790,
  828-830, 872-877, (~916), 959-964. A plain `cudaMalloc` fails at the same offsets as the chunk order:
  +964..+1918 MB versus +964..+1928 MB. The same physical layout sits behind both kinds of allocation.
- **Virtual offset is NOT stable.** The band's chunk index moved 578, 570, 564, 524, 482 between runs a minute
  or two apart, because Maya (restarted at 08:03:45 by a parallel tuning session) grew from 12,116 to 12,410
  MiB under it. Any "skip offsets X to Y" scheme silently breaks whenever anything below the band changes size.
  **Placement only holds up if you test what you actually received and hold it.** That is what `bandmap hold`
  and `bandmap span` do.
- **The physical place agrees with the earlier map.** Maya's ~12.3 GB, plus a context of about 0.15 GB, plus
  the band at +0.96..+1.93 GB comes to about 13.4..14.4 GB from the bottom of the card. The earlier whole-card
  test (`data/rig-maya-trace`, same boot, about 06:40) found +13,268..+14,238 MB. **I have not tested whether
  it stays put across reboots** (that needs the window).
- **Allocation order matters.** The holder only works if it allocates while the band is still free. The band
  starts about 13.3 GB up and allocation is bottom-up, so anything that has already filled the card past 13 GB
  is sitting on it. The holder must start first, on an empty card, before Maya, and Maya must stop if the
  holder stops.
- **The bad set is a stable band with flickering parts.** 1-pass detection found 29 chunks. 6-pass runs found
  48, 53, 52, 51, 46 and 51 chunks, always inside the same about 966 MiB span. Whole clusters can go unseen in
  one 6-pass run: the ~916 cluster in two runs, the ~828 cluster in one. The 10-pass verify after the cluster
  hold (with that cluster not held) did not catch it either, so these clusters flip only now and then.

## Q2: can the driver or NVML retire or remap pages?

**No, not on this card with the stock driver.**
- `nvidia-smi -q -d ROW_REMAPPER,PAGE_RETIREMENT,ECC` on GPU 0 shows ECC Mode N/A, Retired Pages N/A and
  Remapped Rows N/A. Row remapping exists only on Ampere and later datacenter parts. Page retirement needs ECC
  and an InfoROM retired-page list, and this GeForce has neither.
- Driver 595.91.07, the **open** kernel module (`Dual MIT/GPL`), with **GSP firmware 595.91.07** in use
  (`EnableGpuFirmware: 18`). I read the open source (github NVIDIA/open-gpu-kernel-modules, tag 615.78.08):
  - The kernel heap's blacklist is filled only from the InfoROM through `NV2080_CTRL_CMD_FB_GET_OFFLINED_PAGES`,
    and only when page retirement is supported:
    `src/nvidia/src/kernel/gpu/mem_mgr/arch/maxwell/mem_mgr_gm107.c:1667`. That support comes from GSP:
    `src/nvidia/src/kernel/gpu/gpu_gspclient.c:119-125` returns `bPageRetirementSupported` from GSP's static
    info. It is false on a non-ECC GeForce.
  - The registry key `RmBlackListAddresses` (`src/nvidia/interface/nvrm_registry.h:1044`, binary list of
    addresses) has **no consumer anywhere in the open kernel-side source**. It may be read inside the closed
    GSP firmware; I have not verified that. `NVreg_RegistryDwords` can only pass DWORDs, not this binary key.
  - `OverrideFbSize` (`nvrm_registry.h:923`, used at `mem_mgr.c:204`) could only shrink the card from the top.
    The band is at about 13.3 GB, so it would be no better than today's cap.
  - `nvidia-smi --gpu-reset` (needs sudo and an idle GPU) only resets the GPU. It retires and remaps nothing.
- A patched open kernel module could, in principle, fill the heap blacklist with fixed physical addresses
  regardless of the retirement flag. That needs sudo, a rebuild of the module for every driver update, and
  physical addresses that user space cannot see. It is not worth it next to the user-space hold.

## Q3: can Maya skip the range itself?

- **Skipping an offset in the pool: not recommended.** As Q1 shows, the virtual offset of the band within
  Maya's pool depends on everything allocated before it. Maya allocates the main pool in one
  `cudaMalloc(&one, main_bytes)`, splitting into segments only if that fails (`src/core/glm_fast_path.cu:1378`
  on the rig's v1.1.36 checkout; branch `trace-nan` 5412e2e adds the per-device `STRATA_GLM_VRAM_GB`).
- **In-process quarantine: recommended if Maya should own the fix.** Add an opt-in step, for example
  `STRATA_GLM_QUARANTINE=1,0` (one flag per device), at device init before any large allocation. It runs the
  `bandmap` logic: 2 MiB `cuMemCreate` chunks over all free VRAM, the patterns random, ones and inverted
  random, N passes, keep the chunks from the first bad to the last ± margin, release the rest, log what is
  held, and keep the handles for the life of the process. Maya's later plain `cudaMalloc`s then avoid those
  pages; this test shows that (the 7.2 GB and 8.05 GB single-`cudaMalloc` runs straddling the held span both
  came out clean). It saves the separate helper's ~158 MiB context and removes the need to order the helper
  before Maya. I did not write it: the brief allows a branch, but testing it needs a Maya restart, which is
  outside this task. It would be a separate ship task, about 80 lines in `glm_fast_path.cu`, plus `-lcuda`.

## Q4: map of the band, and the hold test (all beside the running Maya)

Tool: `/tmp/bandmap` on the rig, built from `/tmp/bandmap.cu`
(`nvcc -O2 -arch=sm_75 -o /tmp/bandmap /tmp/bandmap.cu -lcuda`).
Usage: `bandmap <dev> <keep_free_mb> <passes> <hold_ms> test|hold|span [margin_chunks] [chunk_mb]`. Each pass
fills with seeded random, ones and inverted random, holds for `hold_ms`, then checks. Zeros is skipped because
it has never failed. It prints per-pattern lane counts, a `BAD chunk` line with the 64 KiB pages hit, and a
`SUMMARY`. In hold or span mode it prints `HOLD:` and then holds until SIGINT or SIGTERM. Every run left
1,024 MB free beside Maya.

Runs, all on GPU 0, all UTC on 2026-10-29. GPU busy time totalled about **2.8 minutes**, in 7 bursts:

| Time | Command | Result |
|---|---|---|
| 08:04:07-08:04:12 | `bandmap 0 1024 1 1000 test` | 8,702 MB tested; 29 bad chunks, index 578..1067 (+1156..+2134 MB); lanes [0 11527 0 11568] (random) |
| 08:04:20-08:04:28 | `bandmap 0 1024 6 300 test` | 48 bad chunks, 570..1052; "ones" also flips a few bits (cleared), mostly lanes 1/3 |
| 08:04:50-08:05:08 | `bandmap 0 1024 6 300 hold 3` + fresh `bandmap 0 1024 6 300 test` | holder kept 121 chunks = 242 MiB; **verify 8,172 MB, 0 bad** |
| 08:05:20-08:05:42 | `bandmap 0 1024 6 300 hold 0` + fresh `bandmap 0 1024 10 300 test` | held 52 chunks = 104 MiB; **verify 8,228 MB, 10 passes, 0 bad** |
| 08:06:40-08:07:38 | `bandmap 0 1024 6 300 span 16` + fresh `bandmap 0 1024 8 2000 test` | held 515 chunks = 1,030 MiB; **verify 7,220 MB, 8 passes x 2 s hold, 0 bad** |
| 08:08:37-08:09:22 | `span 16` and then `hold 2`, each + fresh `/tmp/vramtest 0 2 1024` (one plain `cudaMalloc`) | span: 7,221 MB, all 6 patterns 0 bad; hold 2 (198 MiB): 8,053 MB, 0 bad |
| 08:09:28-08:09:41 | control, no holder: `/tmp/vramtest 0 2 1024` | 8,409 MB: ones 26 bad words; random **8,959 and 8,153 bad words at +964..+1918 MB**, lanes [0 x 0 x] |

The helper's context costs about 158 MiB: free memory with no holder (9,591) = held (198) + the holder's own
free reading (9,235) + 158. Maya (PID 115240, `STRATA_GLM_VRAM_GB=12.5,0`) kept serving throughout:
`maya-s24-t1.log` shows its full capped CUDA0 pool (6.50 GB, 44-49 slots per layer), finished requests and no
errors. It saw 21.25 GB free at split time, before my first burst. I stopped every holder (checked:
`pgrep -af ^/tmp/bandmap` shows none).

## Risks

- **The bad set may grow or move.** The band is a fault in a physical part (one byte lane of one memory chip,
  judging by lanes 1/3 only); it may get worse. The holder must test on every boot and fail closed: if it
  finds bad chunks outside the expected about 1 GB span, or more than a threshold such as 120, it should exit
  non-zero so Maya falls back to the 12.5 GB cap.
- **Intermittent clusters.** Whole clusters escape a 6-pass run. Span plus margin covers the ones seen so far;
  a test on an empty card with more passes (below) is the right way to size it.
- **If the holder dies, the band is free again.** A running Maya keeps its memory, but the next thing that
  fills the card lands on the band. Maya's unit must `BindsTo=` the holder (or Maya must quarantine in-process).
- **Maya's own region is not tested.** I could not test the lower 12.4 GB while Maya held it. The earlier
  whole-card test and the clean capped run suggest it is fine; the window test will confirm.
- **Other users of GPU 0** (the vLLM Qwen unit, if it ever runs there) would see about 1.2 GB less free, which
  matters for `gpu_memory_utilization`, a fraction of the total.

## Whole-card window plan (needs Maya stopped; about 5 minutes of GPU time)

1. Stop Maya. Check GPU 0 is empty: `nvidia-smi --query-compute-apps=pid --format=csv`.
2. Map the whole card:
   `timeout 300 /tmp/bandmap 0 256 20 1000 test > /tmp/bandmap-wc0.log 2>&1` (about 21.7 GB, 20 passes,
   about 90 s). Expect bad chunks only at about +13.2..+14.3 GB and none below 13 GB. Run the same on GPU 1 as
   a control.
3. Hold and verify: `setsid nohup /tmp/bandmap 0 256 10 300 span 16 > /tmp/bandmap-hold.log 2>&1 < /dev/null &`,
   wait for `^HOLD`, then `timeout 300 /tmp/bandmap 0 256 20 1000 test` (expect 0 bad) and
   `timeout 300 /tmp/vramtest 0 30 256` (expect 0 bad words).
4. With the holder still running, start Maya with CUDA0 uncapped (`STRATA_GLM_VRAM_GB=0,0` or unset; with the
   `trace-nan` build `0` means no cap for that card) and `STRATA_GLM_NAN_TRACE=1`. Run a few chats. Expect no
   nan-guard lines and a CUDA0 pool of about 14 GB.
5. To check reboot stability: reboot, rerun step 2 and compare where the span starts. Rebuild first, since
   `/tmp` does not survive a reboot (see below).
6. For production, install the tool durably, for example `~/tools/bandmap`, not `/tmp`. Add
   a systemd **user** unit `gpu0-quarantine.service`
   (`ExecStart=~/tools/bandmap 0 256 10 300 span 16`, `Before=` the Maya unit). Have its
   ExecStartPost wait for `HOLD:` in the log. Give the Maya unit `Requires=`/`BindsTo=` it. No sudo is needed.

## Recommendation / what should ship

- After the window test passes: deploy the span-hold helper as above and remove CUDA0's 12.5 GB cap. That
  brings GPU 0 from 12.5 GB to about 20.8 GB.
- Better later: the in-Maya quarantine patch (Q3). It saves the helper context and the ordering problem.
- None of this replaces the hardware remedy (replacing or RMA-ing the card). It is a workaround with the risks
  above.

## Not verified

Whether the band stays in place across reboots; whether Maya's lower 12.4 GB is clean; whether GSP firmware
reads `RmBlackListAddresses`; how the band behaves under temperature or clocks (clocks untouched).

## Source

See `tools/bandmap.cu`.
