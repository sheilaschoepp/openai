#!/bin/bash

for((i=0;i<=48;i+=2))
do
  sbatch s$i.sh
done