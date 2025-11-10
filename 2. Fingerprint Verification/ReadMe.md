# Fingerprint Module (Detection → Enhancement → Matching)

## Structure
- tools/fp_rank_gallery.py   # ranking/matching (ORB + Lowe ratio)
- vis_fp/                    # data: probe_01 + gallery_* folders
- rank.csv                   # example output
- fingerprint_env.yml        # conda env spec

## Quickstart
conda env create -f fingerprint_env.yml -n fpmod
conda activate fpmod

# Run ranking (change probe_01 if needed)
python tools/fp_rank_gallery.py "vis_fp/probe_01" \
  --gallery_dir "vis_fp" \
  --kind auto --agg mean --features 2000 --ratio 0.9 --topk 10 \
  --csv rank.csv

# View results
head -20 rank.csv
