#!/bin/bash

SCRIPT_DIR=`dirname $0`
cd $SCRIPT_DIR
echo $SCRIPT_DIR
pwd

#$ -l f_node=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -cwd
##$ -o std/std0/out
##$ -e std/std0/err

pwd 

## Initialize and load module
source /etc/profile.d/modules.sh
module load python/3.6.5
module load cuda/10.1.105
module load cudnn/7.6

source ~/venv/pytorch/bin/activate

echo $1
echo $2
source ~/fairseq/workplace/script/$1 $2
