#!/bin/bash

for((i=99;i>=51;i-=3))
do
  sbatch s$i.sh
done