#!/bin/bash

# 1, 48, 49 done
# 0 on melco

for((i=46;i>=2;i-=2))
do
  sbatch s$i.sh
done