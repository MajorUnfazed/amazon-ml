"""Test domain stripping, core name, and blocking keys on Ground Truth."""

import os
import re

train_dir = os.path.join("dataset", "train")

RE_DOMAIN = re.compile(r"(?:https?://)?(?:www\.)?([a-zA-Z0-9\-]+)\.(?:com|org|net|in|co\.in|fr|io|info|biz|co)\b", re.IGNORECASE)

def clean_name_advanced(name: str) -> str:
    # Check if name is a domain
    m = RE_DOMAIN.search(name)
    if m:
        name = m.group(1)
    # clean punctuation
    clean = re.sub(r"[^\w\s]", " ", name).lower()
    return " ".join(clean.split())

# Check on first 5,000 GT matches
matched_samples = []
with open(os.path.join(train_dir, "train_ground_truth.tsv"), "r", encoding="utf-8", errors="ignore") as f:
    next(f)
    for line in f:
        p = line.rstrip("\r\n").split("\t")
        if len(p) > 1 and p[1].strip():
            matched_samples.append((p[0], [x.strip() for x in p[1].split(",") if x.strip()]))
            if len(matched_samples) >= 5000:
                break

needed_s1 = {x[0] for x in matched_samples}
needed_partners = {pid for x in matched_samples for pid in x[1]}

s1_names = {}
s1_addrs = {}
with open(os.path.join(train_dir, "train_source1.tsv"), "r", encoding="utf-8", errors="ignore") as f:
    next(f)
    for line in f:
        p = line.rstrip("\r\n").split("\t")
        if p[0] in needed_s1:
            s1_names[p[0]] = clean_name_advanced(p[1])
            s1_addrs[p[0]] = p[2].lower()

partner_names = {}
partner_addrs = {}
for fn in ["train_source2.tsv", "train_source3.tsv"]:
    with open(os.path.join(train_dir, fn), "r", encoding="utf-8", errors="ignore") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[0] in needed_partners:
                partner_names[p[0]] = clean_name_advanced(p[1])
                partner_addrs[p[0]] = p[2].lower()

exact_name_match = 0
token_overlap_match = 0
total_links = 0

for s1_id, partners in matched_samples:
    s1_n = s1_names.get(s1_id, "")
    s1_toks = set(s1_n.split())
    for pid in partners:
        p_n = partner_names.get(pid, "")
        p_toks = set(p_n.split())
        total_links += 1

        if s1_n and p_n and (s1_n == p_n or s1_n in p_n or p_n in s1_n):
            exact_name_match += 1
        elif s1_toks and p_toks and (len(s1_toks & p_toks) >= 1):
            token_overlap_match += 1

print(f"Total True Links: {total_links:,}")
print(f"Exact or Substring Name Matches: {exact_name_match:,} ({exact_name_match/total_links:.2%})")
print(f"Shared Token Matches:           {token_overlap_match:,} ({token_overlap_match/total_links:.2%})")
print(f"Combined Coverage on Name:      {exact_name_match + token_overlap_match:,} ({(exact_name_match + token_overlap_match)/total_links:.2%})")
