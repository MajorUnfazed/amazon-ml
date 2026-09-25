"""Inspect the 2.14% remaining links where both name and addr < 70%."""

import os
import random
from rapidfuzz import fuzz

random.seed(42)
train_dir = os.path.join("dataset", "train")

gt_map = {}
with open(os.path.join(train_dir, "train_ground_truth.tsv"), "r", encoding="utf-8", errors="ignore") as f:
    next(f)
    for line in f:
        p = line.rstrip("\r\n").split("\t")
        if len(p) > 1 and p[1].strip():
            gt_map[p[0]] = [x.strip() for x in p[1].split(",") if x.strip()]

sample_s1_keys = set(random.sample(list(gt_map.keys()), 3000))
sample_gt = {k: gt_map[k] for k in sample_s1_keys}
sample_partner_ids = {pid for plist in sample_gt.values() for pid in plist}

sample_s1_records = {}
with open(os.path.join(train_dir, "train_source1.tsv"), "r", encoding="utf-8", errors="ignore") as f:
    next(f)
    for line in f:
        p = line.rstrip("\r\n").split("\t")
        if p[0] in sample_s1_keys:
            sample_s1_records[p[0]] = {"name": p[1], "addr": p[2], "country": p[3]}

sample_partner_records = {}
for fn in ["train_source2.tsv", "train_source3.tsv"]:
    with open(os.path.join(train_dir, fn), "r", encoding="utf-8", errors="ignore") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[0] in sample_partner_ids:
                sample_partner_records[p[0]] = {"name": p[1], "addr": p[2], "country": p[3]}

out_rows = []
for s1_id, partners in sample_gt.items():
    s1 = sample_s1_records.get(s1_id)
    if not s1:
        continue
    for pid in partners:
        p = sample_partner_records.get(pid)
        if not p:
            continue
        n_score = fuzz.token_sort_ratio(s1["name"].lower(), p["name"].lower())
        a_score = fuzz.token_set_ratio(s1["addr"].lower(), p["addr"].lower()) if (s1["addr"] and p["addr"]) else 0
        if n_score < 70 and a_score < 70:
            out_rows.append((s1, p, n_score, a_score))

print(f"Total difficult pairs (<70% both): {len(out_rows)}")
for s1, p, ns, as_ in out_rows[:15]:
    print(f"S1: {s1['name']} | {s1['addr']}")
    print(f"P:  {p['name']} | {p['addr']}")
    print(f"    Name Score: {ns}%, Addr Score: {as_}%")
    print("-" * 60)
