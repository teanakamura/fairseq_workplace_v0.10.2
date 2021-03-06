#!/bin/bash
source ~/venv/fairseq/bin/activate

cd `dirname $0`

ps=(050 080 100)
annt_types=(top all)
tag_types=(tag notag)
inln_types=(sinln stinln none)

ks=(0 1 2 3 4 5)
# tasks=(task1_all task2_keyword_set1 task2_keyword_set2 task2_keyword_set3 task2_keyword_set_all)
tasks=(task1_certain0 task1_certain1 task1_certain2 task2_keyword_set_certain0 task2_keyword_set_certain1 task2_keyword_set_certain2)

for p in ${ps[@]}; do
  for annt_type in ${annt_types[@]}; do
    for inln_type in ${inln_types[@]}; do
      # ./preprocess.bash ./yaml_configs/jijinews/final/css_prob${p}_${annt_type}_${inln_type}.yml
      # ./preprocess.bash -a ./yaml_configs/jijinews/final/css_prob${p}_${annt_type}_${inln_type}.yml
      for task in ${tasks[@]}; do
        ./preprocess.bash -t ./yaml_configs/jijinews/final/css_prob${p}_${annt_type}_${inln_type}/test_baobab_${task}.yml
        ./preprocess.bash -ta ./yaml_configs/jijinews/final/css_prob${p}_${annt_type}_${inln_type}/test_baobab_${task}.yml
      done
    done
  done
done

for tag_type in ${tag_types[@]}; do
  for inln_type in ${inln_types[@]}; do
    # ./preprocess.bash ./yaml_configs/jijinews/final/css_synthetic_all_${tag_type}_${inln_type}.yml
    # ./preprocess.bash -a ./yaml_configs/jijinews/final/css_synthetic_all_${tag_type}_${inln_type}.yml
    for task in ${tasks[@]}; do
      ./preprocess.bash -t ./yaml_configs/jijinews/final/css_synthetic_all_${tag_type}_${inln_type}/test_baobab_${task}.yml
      ./preprocess.bash -ta ./yaml_configs/jijinews/final/css_synthetic_all_${tag_type}_${inln_type}/test_baobab_${task}.yml
    done
  done
done
