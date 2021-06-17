#!/bin/bash

for((i=68;i<=98;i+=3))
do
  sbatch s$i.sh
done