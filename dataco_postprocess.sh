#!/bin/bash
# dataco_postprocess.sh
# Run after all four level-parallel sweep jobs finish.
# Merges the split-level outputs, then regenerates the four canonical CSVs in
# outputs/copulagan_final/.
#
# Expected input dirs (one per level × architecture-group):
#   outputs/dataco_v2_cg_none/     CopulaGAN --levels none   (main + MIA)
#   outputs/dataco_v2_cg_srj/      CopulaGAN --levels strict+reject (main + MIA)
#   outputs/dataco_v2_tc_none/     TVAE+CTGAN --levels none
#   outputs/dataco_v2_tc_srj/      TVAE+CTGAN --levels strict+reject
set -euo pipefail
REPO="/Users/apple/Downloads/CAG_bundle/synthetic-data-generation"
cd "$REPO"

CG_NONE="outputs/dataco_v2_cg_none"
CG_SRJ="outputs/dataco_v2_cg_srj"
TC_NONE="outputs/dataco_v2_tc_none"
TC_SRJ="outputs/dataco_v2_tc_srj"
CG_OUT="outputs/dataco_v2_copulagan"
TC_OUT="outputs/dataco_v2_tvae_ctgan"
FINAL="outputs/copulagan_final"

# ---- guard: all four sweep_long.csv must exist ----
for d in "$CG_NONE" "$CG_SRJ" "$TC_NONE" "$TC_SRJ"; do
    if [ ! -f "$d/sweep_long.csv" ]; then
        echo "ERROR: sweep not finished — $d/sweep_long.csv missing" >&2
        exit 1
    fi
done

mkdir -p "$CG_OUT" "$TC_OUT"

# ---- merge level-split long CSVs ----
echo "Merging CopulaGAN none + strict+reject..."
python3 - <<'PYEOF'
import pandas as pd
cg = pd.concat([
    pd.read_csv("outputs/dataco_v2_cg_none/sweep_long.csv"),
    pd.read_csv("outputs/dataco_v2_cg_srj/sweep_long.csv"),
], ignore_index=True)
cg.to_csv("outputs/dataco_v2_copulagan/sweep_long.csv", index=False)
print(f"  CopulaGAN merged: {len(cg)} rows, "
      f"levels={sorted(cg['level'].unique())}, "
      f"architectures={sorted(cg['architecture'].unique())}, "
      f"seeds={sorted(cg['seed'].unique())}")
PYEOF

echo "Merging TVAE+CTGAN none + strict+reject..."
python3 - <<'PYEOF'
import pandas as pd
tc = pd.concat([
    pd.read_csv("outputs/dataco_v2_tc_none/sweep_long.csv"),
    pd.read_csv("outputs/dataco_v2_tc_srj/sweep_long.csv"),
], ignore_index=True)
tc.to_csv("outputs/dataco_v2_tvae_ctgan/sweep_long.csv", index=False)
print(f"  TVAE/CTGAN merged: {len(tc)} rows, "
      f"levels={sorted(tc['level'].unique())}, "
      f"architectures={sorted(tc['architecture'].unique())}, "
      f"seeds={sorted(tc['seed'].unique())}")
PYEOF

# ---- run the combine + canonical CSV generator ----
python3 dataco_combine_results.py \
    --copulagan "$CG_OUT" \
    --tvae_ctgan "$TC_OUT" \
    --out "$FINAL"

echo ""
echo "Done. Canonical CSVs updated in $FINAL/"
