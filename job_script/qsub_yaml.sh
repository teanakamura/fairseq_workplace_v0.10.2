#!/bin/zsh
#@(#) This script is to easily execute qsub

_usage() {
  echo "Usage:"
  echo "  source qsub.sh -d DATA_BASE_NAME -e EXEC_SCRIPT -c CONFIG_FILE [-o STDOUT_FILE]\n"
  #exit 1
}
echo $0

CURRENT_DIR=`pwd`
SCRIPT_DIR=`dirname $0`
CHECK_d=; CHECK_e=; CHECK_c=; FAIL_OTHER=;
DATA=; STDOUT=; EXE=; CONFIG=;
OPTIND=1
while getopts d:o:e:c: OPT
do
  case ${OPT} in
    d) CHECK_d=true
       DATA=${OPTARG}
       echo "DATA: $DATA"
       ;;
    o) STDOUT=${OPTARG}
       echo "STDOUT: $STDOUT"
       ;;
    e) CHECK_e=true
       EXE=${OPTARG}
       echo "EXE: $EXE"
       ;;
    c) CHECK_c=true
       CONFIG=${OPTARG}
       echo "CONFIG: $CONFIG"
       ;;
    :|\?) FAIL_OTHER=true  # Missing required argument | Invalid option
          _usage $0
          ;;
  esac
done
shift `expr ${OPTIND} - 1`

[[ ${CHECK_d}${CHECK_e}${CHECK_c} != truetruetrue && $FAIL_OTHER != true ]] && _usage $0  # Missing required option
while [ -z $DATA ] || [ ! -d "$SCRIPT_DIR/../data-bin/$DATA" ]; do
  ls "$SCRIPT_DIR/../data-bin"
  read "DATA?Input data base name (-d): "
  echo
done
while [ ! -f "$SCRIPT_DIR/../script/$EXE" ]; do
  ls "$SCRIPT_DIR/../script" | grep -E ".+yaml\.sh" --colour=never
  read "EXE?Input execution file (-e): "
  echo
done
while [ ! -f "$SCRIPT_DIR/../script/yaml_configs/$DATA/$CONFIG" ]; do
  ls "$SCRIPT_DIR/../script/yaml_configs/$DATA" | grep -E ".+\.yml" --colour=never
  read "CONFIG?Input config file (-c): "
  echo
done
if [ -z $STDOUT ]; then
  if echo $EXE | grep 'train' > /dev/null; then
    SUBDIR=tra
  elif echo $EXE | grep 'generate' > /dev/null; then
    SUBDIR=gen
  else
    SUBDIR=`echo $EXE | cut -d '.' -f 1`
  fi
  STDOUT=$DATA/${CONFIG%\.yml}/$SUBDIR
  echo "STDOUT: $STDOUT"
fi

cd $SCRIPT_DIR
JOB_SCRIPT=${JOB_SCRIPT:=~/fairseq_v0.10.2/workplace/job_script/job_yaml.sh}
echo "qsub -o std/$STDOUT $JOB_SCRIPT $EXE $DATA $CONFIG"
qsub -o std/$STDOUT $JOB_SCRIPT $EXE $DATA $CONFIG

cd $CURRENT_DIR
unset DATA STDOUT EXE CONFIG

