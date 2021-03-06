#!/bin/bash

function error () {
  check_exec_shell 'bash'; echo $?

  echo 'error'
}

FLAG_A=false
FLAG_T=false
while getopts ath OPT
do
  case ${OPT} in
    a) FLAG_A=true  ## additional data
       ;;
    t) FLAG_T=true  ## test only
       ;;
    h) usage_exit
  esac
done
shift `expr ${OPTIND} - 1`

source ~/dotfiles/cli/cli_func/check_exec_shell.sh
if check_exec_shell 'bash'; then
  SCRIPT_DIR=`dirname $0`
  source ~/dotfiles/cli/cli_func/hierarchical_load_yaml.bash
  h_load_yaml "${SCRIPT_DIR}/yaml_configs" $1

  EXEC_FILE_PATH=$SCRIPT_DIR/../../fairseq/fairseq_cli
  DATA_DIR=$HOME/Data/$data_name/$data_type
  DEST_DIR=$SCRIPT_DIR/../data-bin/$data_name/${dest_dir:-$data_type}

  OPTIONAL_ARGS=(
    --task translation
    --nwordssrc 50000
    --nwordstgt 50000
    --workers 16
  )
  if $FLAG_A && ! $FLAG_T; then
    DEST_DIR=${DEST_DIR}/${dest_add_dir:-additional_data}
    OPTIONAL_ARGS=(
      "${OPTIONAL_ARGS[@]}"
      --trainpref $DATA_DIR/train
      --validpref $DATA_DIR/val
      --testpref $DATA_DIR/test
      --destdir $DEST_DIR
      --source-lang $lang_add
      --only-source
    )
  elif $FLAG_T && ! $FLAG_A; then
    DICT_DIR=${DEST_DIR}
    DEST_DIR=${DEST_DIR}/${external_test_dir}
    OPTIONAL_ARGS=(
      "${OPTIONAL_ARGS[@]}"
      --testpref $DATA_DIR/test
      --destdir $DEST_DIR
      --srcdict $DICT_DIR/dict.${lang_src}.txt
      --source-lang $lang_src
      --target-lang $lang_tgt
      --joined-dictionary
    )
  elif $FLAG_A && $FLAG_T; then
    DICT_DIR=${DEST_DIR}/${dest_add_dir:-additional_data}
    DEST_DIR=${DEST_DIR}/${external_test_dir}/${dest_add_dir:-additional_data}
    OPTIONAL_ARGS=(
      "${OPTIONAL_ARGS[@]}"
      --testpref $DATA_DIR/test
      --destdir $DEST_DIR
      --srcdict $DICT_DIR/dict.${lang_add}.txt
      --source-lang $lang_add
      --only-source
    )
  else
    OPTIONAL_ARGS=(
      "${OPTIONAL_ARGS[@]}"
      --trainpref $DATA_DIR/train
      --validpref $DATA_DIR/val
      --testpref $DATA_DIR/test
      --destdir $DEST_DIR
      --source-lang $lang_src
      --target-lang $lang_tgt
      --joined-dictionary
    )
  fi

  echo DATA_DIR: $DATA_DIR
  echo DEST_DIR: $DEST_DIR
  echo "python ${OPTIONAL_ARGS[@]}"
  python ${EXEC_FILE_PATH}/preprocess.py ${OPTIONAL_ARGS[@]}
else
  error
fi


# python ${EXEC_FILE_PATH}preprocess.py \
#   --task translation \
#   --source-lang $lang_src \
#   --target-lang $lang_tgt \
#   --trainpref $DATA_DIR/train \
#   --validpref $DATA_DIR/val \
#   --testpref $DATA_DIR/test \
#   --destdir $DEST_DIR \
#   --nwordssrc 50000 \
#   --nwordstgt 50000 \
#   --joined-dictionary \
#   --workers 16 \
#
#
# python ${EXEC_FILE_PATH}preprocess.py \
#   --task translation \
#   --source-lang $lang_add \
#   --trainpref $DATA_DIR/train \
#   --validpref $DATA_DIR/val \
#   --testpref $DATA_DIR/test \
#   --destdir $DEST_ADD_DIR \
#   --nwordssrc 50000 \
#   --nwordstgt 50000 \
#   --workers 16 \
#   --only-source
#
# python ${EXEC_FILE_PATH}preprocess.py \
#   --task translation \
#   --source-lang doc \
#   --target-lang sum \
#   --testpref ${DATA_DIR}/test \
#   --destdir ${DEST_DIR} \
#   --nwordssrc 50000 \
#   --nwordstgt 50000 \
#   --joined-dictionary \
#   --workers 16 \
#   --srcdict ${DICT_DIR}/dict.doc.txt\
#
