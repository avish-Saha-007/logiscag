#!/bin/bash
# dataco_relaunch.sh
# Waits for TC-srj (PID 4177) to finish, then relaunches the three crashed jobs
# (CG-none, CG-srj, TC-none) with the gumbel-NaN-safe pipeline.
#
# Usage: bash dataco_relaunch.sh
set -euo pipefail
REPO="${REPO:-$(git rev-parse --show-toplevel)}"
cd "$REPO"
source .venv/bin/activate

TC_SRJ_PID=4177

echo "Waiting for TC-srj (PID $TC_SRJ_PID) to finish..."
while kill -0 "$TC_SRJ_PID" 2>/dev/null; do
    sleep 30
    echo "  $(date '+%H:%M:%S') — TC-srj still running..."
done
echo "TC-srj finished at $(date '+%H:%M:%S')."

# Verify TC-srj produced output
if [ ! -f "outputs/dataco_v2_tc_srj/sweep_long.csv" ]; then
    echo "WARNING: outputs/dataco_v2_tc_srj/sweep_long.csv not found — TC-srj may have crashed." >&2
    echo "Check outputs/dataco_v2_tc_srj/run.log before proceeding." >&2
    exit 1
fi
echo "TC-srj sweep_long.csv confirmed."

# Clear stale run logs from crashed jobs
for d in dataco_v2_cg_none dataco_v2_cg_srj dataco_v2_tc_none; do
    rm -f "outputs/$d/run.log" "outputs/$d/sweep_long.csv"
done

echo ""
echo "Launching three jobs in parallel..."

# TC-none (TVAE+CTGAN, none) — fastest, finishes ~2h
python privacy_utility_sweep.py \
    --real dataco_canonical_40k.csv \
    --architectures TVAE CTGAN \
    --levels none \
    --seeds 5 \
    --n-synth 10000 \
    --epochs 50 \
    --out outputs/dataco_v2_tc_none \
    >"outputs/dataco_v2_tc_none/run.log" 2>&1 &
TC_NONE_PID=$!
echo "TC-none started (PID $TC_NONE_PID)"

# CG-none (CopulaGAN, none, with MIA) — ~2.5h
python privacy_utility_sweep.py \
    --real dataco_canonical_40k.csv \
    --architectures CopulaGAN \
    --levels none \
    --seeds 5 \
    --n-synth 10000 \
    --epochs 100 \
    --out outputs/dataco_v2_cg_none \
    --mia \
    >"outputs/dataco_v2_cg_none/run.log" 2>&1 &
CG_NONE_PID=$!
echo "CG-none started (PID $CG_NONE_PID)"

# CG-srj (CopulaGAN, strict+reject, with MIA) — ~3-4h
python privacy_utility_sweep.py \
    --real dataco_canonical_40k.csv \
    --architectures CopulaGAN \
    --levels strict+reject \
    --seeds 5 \
    --n-synth 10000 \
    --epochs 100 \
    --out outputs/dataco_v2_cg_srj \
    --mia \
    >"outputs/dataco_v2_cg_srj/run.log" 2>&1 &
CG_SRJ_PID=$!
echo "CG-srj started (PID $CG_SRJ_PID)"

echo ""
echo "All three jobs running. Monitor with:"
echo "  tail -f outputs/dataco_v2_tc_none/run.log"
echo "  tail -f outputs/dataco_v2_cg_none/run.log"
echo "  tail -f outputs/dataco_v2_cg_srj/run.log"
echo ""
echo "When all three finish, run: bash dataco_postprocess.sh"
