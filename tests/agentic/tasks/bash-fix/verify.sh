#!/bin/bash
# runs in the task's work directory after the run; exit 0 = pass
out=$(bash report.sh | sed 's/^ *//')
[ "$out" = "$(printf '5 a\n3 b\n2 c')" ] || exit 1
