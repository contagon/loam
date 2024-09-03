#!/bin/bash

array=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
length=$1
echo "Length: $length"

file="results/$length.csv"
touch $file
echo "length error percent_edge percent_planar mean_edge mean_planar" > $file

for e in "${array[@]}"; do
    for p in "${array[@]}"; do
        echo -e "\e[36mEdge: $e, Planar: $p\e[0m"
        python run.py newer_short/ -kr 10 --length $length --percent_edge $e --percent_planar $p >> $file
    done
done