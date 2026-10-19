#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs
python src/e1_rope_publication_v1.py --smoke --output outputs/e1_rope_smoke.json
python src/b_qkv_output_norm_rope.py --smoke --out outputs/b_qkv_smoke.json
python src/e7_rope_aa.py --batches 1 --sizes 128 --rounds 1 --samples 3 \
  --out outputs/e7_smoke.json
