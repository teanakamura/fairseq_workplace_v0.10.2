#!/bin/bash

echo $JOB_ID

SCRIPT_DIR=`dirname $0`
SH_UTILS_DIR=$SCRIPT_DIR/../sh_utils
YAML_ROOT_PATH=$SCRIPT_DIR/yaml_configs

source $SCRIPT_DIR/sh_env.sh

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

# #CUDA_VISIBLE_DEVICES=0,2,3 \
declare -a OPTIONAL_ARGS
OPTIONAL_ARGS=(
   --save-dir ${SAVE_DIR}
   --arch ${CONF[model_arch]}
   --task ${CONF[model_task]}
     --truncate-source
   --ddp-backend no_c10d
   --criterion ${CONF[model_criterion]}
   --optimizer adam
      --adam-betas "(0.9,0.98)"
   --lr ${CONF[lr_init]}
   --lr-scheduler inverse_sqrt
   --min-lr ${CONF[lr_min]}
   --warmup-updates ${CONF[warmup_updates]}
   --warmup-init-lr ${CONF[warmup_init_lr]}
   --dropout ${CONF[reg_dropout]}
   --weight-decay ${CONF[reg_weight_decay]}
   --decoder-learned-pos
   --encoder-learned-pos
   --log-format simple
   --log-interval 1000
   --fixed-validation-seed 7
   --max-tokens ${CONF[io_max_tokens]}
   --update-freq ${CONF[update_freq]}
   --save-interval-updates 20000
   --clip-norm ${CONF[clip_norm]}
   --max-update ${CONF[update_max]}
   --max-epoch ${CONF[max_epoch]}
   --max-update ${CONF[update_max]}
   --skip-invalid-size-inputs-valid-test
   --user-dir ${USER_DIR}
   --keep-last-epochs ${CONF[keep_last_epochs]}
   --truncate-source
   --tensorboard-logdir ${TENSORBOARD_DIR}
   --source-lang ${CONF[lang_src]}
   --target-lang ${CONF[lang_tgt]}
)
      # --max-source-positions ${CONF[io_max_src]}
      # --max-target-positions ${CONF[io_max_tgt]}
   # --max-sentences ${CONF[io_max_sentences]}

BOOLEAN_OPTIONS=(fp16 cpu reset_optimizer share_all_embeddings)
for option in ${BOOLEAN_OPTIONS[@]}; do
  if ${CONF[$option]}; then
    OPTIONAL_ARGS+=(--${option//_/-})
  fi
done
# if ${CONF[fp16]}; then
#   OPTIONAL_ARGS+=('--fp16')
# fi
# if ${CONF[cpu]}; then
#   OPTIONAL_ARGS+=('--cpu')
# fi
# if ${CONF[reset_optimizer]}; then
#   OPTIONAL_ARGS+=(OPTIONAL_ARGS'--reset-optimizer')
# fi
PARAM_OPTIONS=(seed fp16_scale_tolerance add_dir add_lang)
for option in ${PARAM_OPTIONS[@]}; do
  if [ -n "${CONF[$option]}" ]; then
    OPTIONAL_ARGS+=(--${option//_/-})
    OPTIONAL_ARGS+=(${CONF[$option]})
  fi
done
# if [ -n "${CONF[seed]}" ]; then
#   OPTIONAL_ARGS+=('--seed')
#   OPTIONAL_ARGS+=(${CONF[seed]})
# fi
if [ -n "${CONF[model_criterion_label_smoothing]}" ]; then
  OPTIONAL_ARGS+=('--label-smoothing')
  OPTIONAL_ARGS+=(${CONF[model_criterion_label_smoothing]})
fi
# if [ -n "${CONF[additional_data]}" ]; then
#   OPTIONAL_ARGS+=('--additional-data')
#   OPTIONAL_ARGS+=("${FAIRSEQ_ROOT}/${CONF[additional_data]}")
# fi

# for elem in ${OPTIONAL_ARGS[@]}; do
#   echo $elem
# done
echo "python ${EXEC_FILE_PATH}train.py ${DATA_DIR} ${OPTIONAL_ARGS[@]}"
python ${EXEC_FILE_PATH}train.py ${DATA_DIR} ${OPTIONAL_ARGS[@]}
unset CONF
