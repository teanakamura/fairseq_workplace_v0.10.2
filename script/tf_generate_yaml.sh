#!/bin/bash

echo $JOB_ID

SCRIPT_DIR=`dirname $0`
SH_UTILS_DIR=$SCRIPT_DIR/../sh_utils
YAML_ROOT_PATH=$SCRIPT_DIR/yaml_configs

declare -A CONF
eval CONF=(`$SH_UTILS_DIR/load_yaml.sh $YAML_ROOT_PATH $1/$2`)

for k in ${!CONF[@]}; do
  echo $k: ${CONF[$k]}
done

ENV_FILE=${FAIRSEQ_ROOT}/workplace/script/env_yaml
source ${ENV_FILE}
echo EXEC_FILE_PATH: ${EXEC_FILE_PATH}
echo DATA_DIR: ${DATA_DIR}
echo SAVE_DIR: ${SAVE_DIR}
echo USER_DIR: ${USER_DIR}
echo TENSORBOARD_DIR: ${TENSORBOARD_DIR}

if [[ ${CONF[data]: -7} == subword ]]; then
  SYSTEM=system_output-subword.txt
  REFERENCE=reference-subword.txt
else
  SYSTEM=system_output.txt
  REFERENCE=reference.txt
fi

mkdir -p ${OUT_DIR}

#CUDA_VISIBLE_DEVICES=0,2,3 \
declare -a OPTIONAL_ARGS
OPTIONAL_ARGS=(
   --gen-subset ${CONF[gen_subset]}
   --path ${SAVE_FILE}
   --beam ${CONF[beam]}
   --task ${CONF[model_task]}
      --iter-decode-max-iter 30
      --iter-decode-eos-penalty 1
   --max-tokens ${CONF[io_max_tokens]}
   --skip-invalid-size-inputs-valid-test
   --max-source-positions ${CONF[io_max_src]}
   --min-len 5
   --system ${OUT_DIR}/${SYSTEM}
   --reference ${OUT_DIR}/${REFERENCE}
   --truncate-source
   --user-dir ${USER_DIR}
   --source-lang ${CONF[lang_src]}
   --target-lang ${CONF[lang_tgt]}
)
BOOLEAN_OPTIONS=(fp16 cpu reset_optimizer)
for option in ${BOOLEAN_OPTIONS[@]}; do
  if ${CONF[$option]}; then
    OPTIONAL_ARGS+=(--${option//_/-})
  fi
done
# if ${CONF[fp16]}; then
#   OPTIONAL_ARGS+='--fp16'
# fi
# if ${CONF[cpu]}; then
#   OPTIONAL_ARGS+='--cpu'
# fi
# if ${CONF[reset_optimizer]}; then
#   OPTIONAL_ARGS+='--reset-optimizer'
# fi
# if [ -n "${CONF[additional_data]}" ]; then
#   echo "${CONF[additional_data]}"
#   OPTIONAL_ARGS+=(--additional-data)
#   OPTIONAL_ARGS+=(${FAIRSEQ_ROOT}/${CONF[additional_data]})
# fi
PARAM_OPTIONS=(seed fp16_scale_tolerance add_dir add_lang)
for option in ${PARAM_OPTIONS[@]}; do
  if [ -n "${CONF[$option]}" ]; then
    OPTIONAL_ARGS+=(--${option//_/-})
    OPTIONAL_ARGS+=(${CONF[$option]})
  fi
done

# if [ -n "${CONF[model_criterion_label_smoothing]}" ]; then
#   OPTIONAL_ARGS+=('--label-smoothing')
#   OPTIONAL_ARGS+=(${CONF[model_criterion_label_smoothing]})
# fi

echo "python ${EXEC_GEN_FILE_PATH}generate.py ${GEN_DATA_DIR} ${OPTIONAL_ARGS[@]}"
python ${EXEC_GEN_FILE_PATH}generate.py ${GEN_DATA_DIR} ${OPTIONAL_ARGS[@]}
unset CONF

if [[ ${CONF[data]: -7} = subword ]]; then
  CURRENT_DIR=`pwd`
  cd $OUT_DIR
  sed -r 's/(@@ )|(@@ ?$)//g' system_output-subword.txt > system_output.txt
  sed -r 's/(@@ )|(@@ ?$)//g' reference-subword.txt > reference.txt
  cd ${CURRENT_DIR}
fi
