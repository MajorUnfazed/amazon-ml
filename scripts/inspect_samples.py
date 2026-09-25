import os

train_dir = os.path.join("dataset", "train")

gt_matches = []
with open(os.path.join(train_dir, "train_ground_truth.tsv"), "r", encoding="utf-8", errors="ignore") as f:
    next(f)
    for line in f:
        p = line.rstrip("\r\n").split("\t")
        if len(p) > 1 and p[1].strip():
            gt_matches.append((p[0], [x.strip() for x in p[1].split(",") if x.strip()]))
            if len(gt_matches) >= 20:
                break

needed_s1 = {x[0] for x in gt_matches}
needed_partners = {pid for x in gt_matches for pid in x[1]}

s1_data = {}
with open(os.path.join(train_dir, "train_source1.tsv"), "r", encoding="utf-8", errors="ignore") as f:
    next(f)
    for line in f:
        p = line.rstrip("\r\n").split("\t")
        if p[0] in needed_s1:
            s1_data[p[0]] = p

p_data = {}
for fn in ["train_source2.tsv", "train_source3.tsv"]:
    with open(os.path.join(train_dir, fn), "r", encoding="utf-8", errors="ignore") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[0] in needed_partners:
                p_data[p[0]] = p

with open("sample_inspection.txt", "w", encoding="utf-8") as out:
    for s1_id, partners in gt_matches:
        s1 = s1_data.get(s1_id, [s1_id, "UNK", "UNK", "UNK"])
        out.write(f"S1: [{s1[0]}] ({s1[3]})\n")
        out.write(f"    Name:    {s1[1]}\n")
        out.write(f"    Address: {s1[2]}\n")
        out.write(f"  Matches ({len(partners)}):\n")
        for pid in partners:
            p = p_data.get(pid, [pid, "UNK", "UNK", "UNK"])
            out.write(f"    -> [{p[0]}] ({p[3]})\n")
            out.write(f"       Name:    {p[1]}\n")
            out.write(f"       Address: {p[2]}\n")
        out.write("-" * 80 + "\n")

print("Saved sample inspection to sample_inspection.txt")
