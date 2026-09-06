#!/usr/bin/env bash
# Run a Python script with multiple parameter sets using uv run.
# Edit SCRIPT and PARAMS below, then: chmod +x batch_run.sh && ./batch_run.sh

SCRIPT="swan"

PARAMS=(
  
    #"test/Si2/Si2.cif --output-dir test_EBR_iso+hyb  --auto-nk-grid --hybridize-on-site --EBR --input-dir test_min_shells_full"
    #"test/Al/Al.cif  --output-dir test_min_shells_full  --auto-nk-grid --hybridize-on-site "
    #"test/AsGa/AsGa.cif  --output-dir test_EBR_iso+hyb  --auto-nk-grid --hybridize-on-site --EBR --input-dir test_min_shells_full --K 1.5 --alphabet isotypic+hyb"
    #"test/ClNa/ClNa.cif  --output-dir test_EBR_iso+hyb  --auto-nk-grid --hybridize-on-site --EBR --input-dir test_min_shells_full  --K 1.5 --alphabet isotypic+hyb"
    #"test/Cu/Cu.cif --output-dir test_min_shells_full  --auto-nk-grid --hybridize-on-site "
    #"test/Ga2N2/Ga2N2.cif --output-dir test_EBR_iso+hyb --auto-nk-grid --hybridize-on-site --EBR --input-dir test_min_shells_full --K 1.5 --alphabet isotypic+hyb"
    #"test/K/K.cif --output-dir test_new_wb    --auto-nk-grid --hybridize-on-site"
    #"test/MgO/MgO.cif --output-dir test_min_shells_full   --auto-nk-grid --K 1.5 --hybridize-on-site "
    #"test/V/V.cif --output-dir test_min_shells_full  --auto-nk-grid --K 1.5 --hybridize-on-site"
    #"test/W/W.cif --output-dir test_min_shells_full  --auto-nk-grid --K 1.7 --hybridize-on-site "
    #"test/SZn/SZn.cif --output-dir test_EBR_iso+hyb --auto-nk-grid --hybridize-on-site --EBR --input-dir test_min_shells_full --alphabet isotypic+hyb"
    #"SrTiO3.cif --output-dir test_EBR_iso+hyb  --auto-nk-grid --hybridize-on-site --EBR --input-dir test_min_shells_full --K 1.5 --alphabet isotypic+hyb"
    #"MnTe.cif --output-dir test_EBR_iso+hyb  --auto-nk-grid --hybridize-on-site --EBR --input-dir test_min_shells_full --K 1.5 --alphabet isotypic+hyb"

    #"BaTiO3.cif --output-dir test_EBR_iso+hyb  --auto-nk-grid --hybridize-on-site --EBR --input-dir test_min_shells --K 1.5 --alphabet isotypic+hyb"
    #"zhang_cifs/mp-29208.cif --output-dir test_EBR_iso+hyb  --auto-nk-grid --hybridize-on-site --EBR --input-dir cluster_results --K 1.5 --alphabet isotypic+hyb"

    "testset/mp-1079182.cif --output-dir test_EBR_iso+hyb_metals  --auto-nk-grid --hybridize-on-site --EBR --input-dir cluster_results --K 1.5 --alphabet isotypic+hyb"
    "testset/mp-20554.cif --output-dir test_EBR_iso+hyb_metals  --auto-nk-grid --hybridize-on-site --EBR --input-dir cluster_results --K 1.5 --alphabet isotypic+hyb"
    "testset/mp-1080121.cif --output-dir test_EBR_iso+hyb_metals  --auto-nk-grid --hybridize-on-site --EBR --input-dir cluster_results --K 1.5 --alphabet isotypic+hyb"
    "testset/mp-2648.cif --output-dir test_EBR_iso+hyb_metals  --auto-nk-grid --hybridize-on-site --EBR --input-dir cluster_results --K 1.5 --alphabet isotypic+hyb"


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
    LOG="$LOG_DIR/${TIMESTAMP}_run_${IDX}.log"

    echo -n "[$IDX/$TOTAL] mpirun -np 14 uv run $SCRIPT $ARGS ... "

    # shellcheck disable=SC2086
    if mpirun -np 14 uv run "$SCRIPT" $ARGS > "$LOG" 2>&1; then
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