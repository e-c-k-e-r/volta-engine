#!/bin/bash
# three most frequent words of words.txt
tr -s ' ' '\n' < words.txt | sort | uniq -c | sort -n | head -n 3
