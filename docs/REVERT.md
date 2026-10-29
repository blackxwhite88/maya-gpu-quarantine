# Reverting / moving to new hardware

1. Stop Maya, then stop the holder (`kill` the `bandmap` process or `systemctl --user stop gpu0-quarantine`). Nothing on the card or in the driver was changed; no sudo, no clock pinning, no driver settings.
2. Remove `systemd/gpu0-quarantine.service` from `~/.config/systemd/user/` and `systemctl --user daemon-reload`.
3. If the faulty card is replaced, rerun `bandmap 0 256 20 1000 test` on an empty card. Zero bad chunks means the quarantine is no longer needed.
4. Maya config needs no change beyond using the whole card (`config/maya-s24-prod.json` has no per-card cap).
5. Models live on `/dev/shm` for speed; copy `Maya-S24/` to disk for persistence and update `--glm-pack`, `tokenizer` and vision paths in the config.
