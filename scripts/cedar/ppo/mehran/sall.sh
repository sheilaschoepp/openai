#!/bin/bash

for((i=51;i<=99;i+=3))
do
  sbatch s$i.sh
done