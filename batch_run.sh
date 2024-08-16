#!/bin/bash

array=(0.1 0.5 1 5 10 50 100 500 1000 5000 10000)
length=$1
echo "Length: $length"

file="results/$length.csv"
touch $file
echo "length error threshold_edge threshold_planar mean_edge mean_planar" > $file

for e in "${array[@]}"; do
    for p in "${array[@]}"; do
        echo -e "\e[36mEdge: $e, Planar: $p\e[0m"
        python run.py newer_short/ -kr 10 --length $length --threshold_edge $e --threshold_planar $p >> $file
    done
done