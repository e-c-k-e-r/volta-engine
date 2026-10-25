#!/bin/bash
# runs in the task's work directory after the run; exit 0 = pass
python3 -m unittest -q test_textutil >/dev/null 2>&1 || exit 1
grep -q "slugify" test_textutil.py || exit 1
python3 hidden_slugify.py
