#!/bin/sh
# Start Maya with the quarantine already holding. Recipe: stock production build + dual-link prefill only.
MAYA_DIR=${MAYA_DIR:?set MAYA_DIR}
CFG=${CFG:-$MAYA_DIR/maya-s24-prod.json}
grep -q "^HOLD" "${HOLD_LOG:-/tmp/bandmap-hold.log}" || { echo "quarantine not holding; start bandmap first" >&2; exit 1; }
cd "$MAYA_DIR" && exec env MAYA_GLM_DUAL_PREFILL=1 .venv/bin/python maya.py --config "$CFG" --port 8080
