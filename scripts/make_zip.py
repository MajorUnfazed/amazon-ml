#!/usr/bin/env python3
"""
Submission Packager for Amazon ML Challenge: Business Entity Resolution.
Builds <team_name>_submission.zip matching the required competition structure:

<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
"""

import os
import sys
import zipfile
import argparse

def build_submission_zip(team_name: str, root_dir: str = "."):
    zip_filename = f"{team_name}_submission.zip"
    zip_filepath = os.path.join(root_dir, zip_filename)

    print(f"Packaging submission archive: {zip_filename}...")

    # Required files and directories
    output_dir = os.path.join(root_dir, "output")
    matching_tsv = os.path.join(output_dir, "matching_results.tsv")
    candidate_tsv = os.path.join(output_dir, "candidate_pairs.tsv")

    code_dir = os.path.join(root_dir, "code", "business_entity_resolution")
    src_dir = os.path.join(code_dir, "src")
    readme_file = os.path.join(code_dir, "README.md")
    req_file = os.path.join(code_dir, "requirements.txt")

    doc_file = os.path.join(root_dir, "Documentation_template.md")
    if not os.path.exists(doc_file):
        doc_file = os.path.join(root_dir, "docs", "Documentation_template.md")

    # Verify presence
    missing = []
    for path, name in [
        (matching_tsv, "output/matching_results.tsv"),
        (candidate_tsv, "output/candidate_pairs.tsv"),
        (src_dir, "code/business_entity_resolution/src/"),
        (readme_file, "code/business_entity_resolution/README.md"),
        (req_file, "code/business_entity_resolution/requirements.txt"),
        (doc_file, "Documentation_template.md"),
    ]:
        if not os.path.exists(path):
            missing.append(name)

    if missing:
        print(f"Error: Missing required submission items:\n" + "\n".join(f"  - {m}" for m in missing))
        print("Please ensure the pipeline has run and generated output files before packaging.")
        sys.exit(1)

    with zipfile.ZipFile(zip_filepath, "w", zipfile.ZIP_DEFLATED) as zipf:
        # 1. output/
        zipf.write(matching_tsv, arcname="output/matching_results.tsv")
        zipf.write(candidate_tsv, arcname="output/candidate_pairs.tsv")
        print("  Added output/matching_results.tsv")
        print("  Added output/candidate_pairs.tsv")

        # 2. code/business_entity_resolution/
        zipf.write(readme_file, arcname="code/business_entity_resolution/README.md")
        zipf.write(req_file, arcname="code/business_entity_resolution/requirements.txt")
        print("  Added code/business_entity_resolution/README.md")
        print("  Added code/business_entity_resolution/requirements.txt")

        # Add all source files recursively
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                if file.endswith((".py", ".json", ".yaml", ".md", ".txt")) and not file.endswith(".pyc"):
                    abs_p = os.path.join(root, file)
                    rel_p = os.path.relpath(abs_p, code_dir)
                    arc_name = f"code/business_entity_resolution/{rel_p.replace(os.sep, '/')}"
                    zipf.write(abs_p, arcname=arc_name)
        print("  Added code/business_entity_resolution/src/ (recursive)")

        # 3. Documentation_template.md
        zipf.write(doc_file, arcname="Documentation_template.md")
        print("  Added Documentation_template.md")

    # Verify created zip contents
    print("\nVerifying created zip package:")
    with zipfile.ZipFile(zip_filepath, "r") as zipf:
        namelist = zipf.namelist()
        for name in sorted(namelist):
            info = zipf.getinfo(name)
            print(f"  {name:<60} ({info.file_size:,} bytes)")

    size_mb = os.path.getsize(zip_filepath) / (1024 * 1024)
    print(f"\nSuccessfully generated {zip_filename} ({size_mb:.2f} MB)")
    print("Package is ready for portal submission.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package submission zip archive")
    parser.add_argument("--team", default="LinkSure", help="Team name for prefix")
    parser.add_argument("--root-dir", default=".", help="Project root directory")
    args = parser.parse_args()

    build_submission_zip(team_name=args.team, root_dir=args.root_dir)
