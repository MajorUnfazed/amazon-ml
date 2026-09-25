#!/bin/bash
# Shell wrapper to invoke python packager
set -e
TEAM="${1:-LinkSure}"
python3 scripts/make_zip.py --team "$TEAM"
