import os
import sys
import pandas as pd

src_dir = os.path.join(os.getcwd(), "code", "business_entity_resolution", "src")
sys.path.insert(0, src_dir)

from normalize.text import normalize_record

# Load first 15 ground truth lines
gt_df = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', nrows=25, dtype=str).fillna('')
pairs = []
for _, row in gt_df.iterrows():
    s1 = row['source1_entity_id']
    m = row['matched_entity_ids']
    if m:
        for p in m.split(','):
            pairs.append((s1, p))

s1_needed = {x[0] for x in pairs}
p_needed = {x[1] for x in pairs}

s1_lookup = {}
with open('dataset/train/train_source1.tsv', 'r', encoding='utf-8', errors='ignore') as f:
    next(f)
    for line in f:
        p = line.rstrip('\r\n').split('\t')
        if p[0] in s1_needed:
            s1_lookup[p[0]] = (p[1], p[2], p[3])

p_lookup = {}
for fn in ['train_source2.tsv', 'train_source3.tsv']:
    with open(os.path.join('dataset', 'train', fn), 'r', encoding='utf-8', errors='ignore') as f:
        next(f)
        for line in f:
            p = line.rstrip('\r\n').split('\t')
            if p[0] in p_needed:
                p_lookup[p[0]] = (p[1], p[2], p[3])

print(f"Loaded {len(s1_lookup)} S1, {len(p_lookup)} P records.")
print("=" * 80)
for s1, p in pairs[:10]:
    if s1 in s1_lookup and p in p_lookup:
        s1_rec = s1_lookup[s1]
        p_rec = p_lookup[p]
        s1_norm = normalize_record(s1_rec[0], s1_rec[1], s1_rec[2])
        p_norm = normalize_record(p_rec[0], p_rec[1], p_rec[2])
        print(f"S1 ({s1}): {s1_rec[0]}  ||  {s1_rec[1]}")
        print(f"P  ({p}): {p_rec[0]}  ||  {p_rec[1]}")
        s1_slug = str(s1_norm['name_core']).replace(' ', '')[:25]
        p_slug = str(p_norm['name_core']).replace(' ', '')[:25]
        print(f"  s1_slug='{s1_slug}', p_slug='{p_slug}' -> Match? {s1_slug == p_slug}")
        s1_toks = set(str(s1_norm['name_clean']).split())
        p_toks = set(str(p_norm['name_clean']).split())
        print(f"  s1_toks={s1_toks}")
        print(f"  p_toks={p_toks}")
        print(f"  Shared name tokens: {s1_toks.intersection(p_toks)}")
        print("-" * 80)
