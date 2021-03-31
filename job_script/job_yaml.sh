#!/bin/bash

SCRIPT_DIR=`dirname $0`
cd $SCRIPT_DIR
pwd

#$ -l rt_F=1
#$ -l h_rt=20:00:00
#$ -j y
#$ -cwd
##$ -p -400
#$ -o std/std0/out
##$ -e std/std0/err

## load jobrc
# source ~/fairseq_v0.10.2/workplace/job_script/jobrc

## Initialize module
source ~/fairseq_v0.10.2/workplace/job_script/module_load.sh

~/fairseq_v0.10.2/workplace/script/$1 $2 $3
