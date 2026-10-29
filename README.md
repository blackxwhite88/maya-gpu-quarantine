# maya-gpu-quarantine

Run [Project Maya](https://github.com/mw00/project-maya) (GLM-5.3-flash) on a pair of modded 22 GB RTX 2080 Ti cards when one card has a **faulty VRAM band**, without giving up the whole 22 GB.

Fork used for this work: <https://github.com/blackxwhite88/project-maya>

## The problem

On a 2×2080 Ti (22 GB mod, NVLink) rig, Maya produced router/NaN errors and garbage on GPU 0 only. A whole-card pattern test showed bad memory in a band roughly 13.3–14.4 GB into GPU 0's allocation range; GPU 1 was clean. Consumer GeForce cards have no ECC, no page retirement and no row remapping, and the stock driver cannot be told to avoid physical pages. The usual workaround is capping that card to ~12.5 GB, which wastes ~9 GB.

## The workaround

`tools/bandmap.cu` splits free VRAM into separate 2 MiB physical allocations (`cuMemCreate`), pattern-tests every chunk, frees the good ones and **keeps holding the bad ones** (plus a margin). A held allocation is physical memory the driver can't hand to anyone else, so a later process (Maya) never receives the bad pages. Verified: with the holder running, a fresh process filling the rest of the card saw zero bit flips; without it, the same test flips ~9,000 words in the same place.

Cost: ~1.0 GB held + ~158 MiB helper context, so ~20.8 GB of GPU 0 is usable instead of 12.5 GB.

```sh
nvcc -O2 -arch=sm_75 -o bandmap tools/bandmap.cu -lcuda
# start BEFORE Maya, on an empty card; it must keep running while Maya runs
setsid nohup ./bandmap 0 256 10 300 span 16 > /tmp/bandmap-hold.log 2>&1 < /dev/null &
grep ^HOLD /tmp/bandmap-hold.log     # "HOLD: holding 521 chunks = 1042 MB"
```

Usage: `bandmap <dev> <keep_free_mb> <passes> <hold_ms> test|hold|span [margin_chunks] [chunk_mb]`. `test` only maps the card; `span` holds every chunk from the first to the last bad one ± margin (recommended, because the bad set flickers between runs). `systemd/gpu0-quarantine.service` is a user unit for a durable setup; make Maya's unit `BindsTo=` it.

Details, measurements and risks: [`docs/GPU0-BAND-ANALYSIS.md`](docs/GPU0-BAND-ANALYSIS.md). Maya settings and results: [`docs/RECIPE.md`](docs/RECIPE.md). Undo / hardware swap: [`docs/REVERT.md`](docs/REVERT.md).

## Caveats

- This is a workaround for failing hardware, not a repair. The band may grow; the holder should be re-run on every boot and Maya should fall back to a capped pool if the bad set leaves the expected span.
- Placement across reboots is **not yet verified**. Virtual offsets are not stable, so always test what you actually received.
- If the holder dies, the band is free again for the next allocation.
- Tested on driver 595.91.07 (open kernel module), CUDA 13.3, sm_75, Maya v1.1.36.
- Unofficial; not affiliated with the Maya project.
