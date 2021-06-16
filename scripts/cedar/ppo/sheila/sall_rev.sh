#!/bin/bash

for((i=48;i>=0;i-=3))
do
  sbatch s$i.sh
done