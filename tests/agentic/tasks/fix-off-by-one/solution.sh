#!/bin/bash
# reference solution, applied in the work directory (self-test only)
sed -i 's/range(len(xs) - n)/range(len(xs) - n + 1)/' stats.py
