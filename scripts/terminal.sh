#!/bin/bash

# Array of ns values to search for
ns_values=("2048" "4096" "1024" "8192")

# Loop through each ns value and search for matching files
for ns in "${ns_values[@]}"; do
    count=$(ls data | grep "ns:${ns}" | wc -l)
    echo "Files with ns:${ns}: $count"
done