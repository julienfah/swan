#!/usr/bin/env bash
# Run a Python script with multiple parameter sets using uv run.
# Edit SCRIPT and PARAMS below, then: chmod +x batch_run.sh && ./batch_run.sh

SCRIPT="main.py"

PARAMS=(
    "test/Si2/Si2.cif"
    #"test/Ag/Ag.cif"
    #"test/Al/Al.cif"
    #"test/AsGa/AsGa.cif"
    "test/ClNa/ClNa.cif"
    #"test/Cu/Cu.cif"
    #"test/Ga2N2/Ga2N2.cif"
    #"test/K/K.cif"
    #"test/MgO/MgO.cif"
    #"test/V/V.cif"
    #"test/W/W.cif"


    
)

# ── Runner (no need to edit below) ───────────────────────────────────────────

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TOTAL=${#PARAMS[@]}
FAILED=()

echo "Starting $TOTAL runs at $(date)"
echo ""

for i in "${!PARAMS[@]}"; do
    IDX=$(( i + 1 ))
    ARGS="${PARAMS[$i]}"
    LOG="$LOG_DIR/run_${IDX}_${TIMESTAMP}.log"

    echo -n "[$IDX/$TOTAL] uv run $SCRIPT $ARGS ... "

    # shellcheck disable=SC2086
    if uv run "$SCRIPT" $ARGS > "$LOG" 2>&1; then
        echo "OK"
    else
        echo "FAILED (see $LOG)"
        FAILED+=("$IDX: $ARGS")
    fi
done

echo ""
echo "Done at $(date) — $((TOTAL - ${#FAILED[@]}))/$TOTAL succeeded."

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "Failed runs:"
    for f in "${FAILED[@]}"; do
        echo "  $f"
    done
fi