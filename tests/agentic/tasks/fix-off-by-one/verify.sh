#!/bin/bash
# runs in the task's work directory after the run; exit 0 = pass
python3 -m unittest -q test_stats 2>&1 | tail -n 3
python3 -m unittest -q test_stats >/dev/null 2>&1 || exit 1
grep -q "assertEqual(moving_average(\[2, 4, 6\], 3), \[4.0\])" test_stats.py || exit 1
