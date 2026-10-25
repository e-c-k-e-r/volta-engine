#!/bin/bash
# runs in the task's work directory after the run; exit 0 = pass
! grep -rq fetch_data --include=*.py --include=*.md . || exit 1
grep -q "def load_records" store.py || exit 1
python3 -m unittest -q test_store >/dev/null 2>&1 || exit 1
python3 cli.py data.jsonl | grep -q "^3 7$" || exit 1
