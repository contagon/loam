#!/bin/bash

edges=(1 10 50 100 10000 100000)
planar=(0.1 10 50 100 500 1000)
length=$1
echo "Length: $length"

file="results/$length.csv"
touch $file
echo "length error threshold_edge threshold_planar mean_edge mean_planar" > $file

for p in "${planar[@]}"; do
    for edge in "${edges[@]}"; do
        echo -e "\e[36mEdge: $edge, Planar: $p\e[0m"
        python run.py newer_short/ -kr 10 --length $length --threshold_edge $edge --threshold_planar $p >> $file
    done
done