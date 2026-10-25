#!/bin/bash
# runs in the task's work directory after the run; exit 0 = pass
[ "$(tr -d ' \n' < answer.txt)" = "next_delay:12" ] || [ "$(tr -d ' \n' < answer.txt)" = "next_delay:12.0" ] || exit 1
