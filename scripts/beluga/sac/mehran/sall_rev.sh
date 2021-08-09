#!/bin/bash

for((i=98;i>=68;i-=2))
do
  sbatch s$i.sh
done
