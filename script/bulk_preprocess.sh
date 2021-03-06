#!/bin/bash
source ~/venv/fairseq/bin/activate

cd `dirname $0`

ps=(010 020 030 050)
ks=(0 1 2 3 4 5)
bases=(task1_free1 task1_free2 task1_free3 task1_all task2_keyword_set1 task2_keyword_set2 task2_keyword_set3 task2_keyword_set_all task1_certain0 task1_certain1 task1_certain2 task2_keyword_set_certain0 task2_keyword_set_certain1 task2_keyword_set_certain2)

for p in ${ps[@]}; do
  # ./preprocess.bash ./yaml_configs/jijinews/css_srcannt${p}/css_srcannt${p}_unigram.yml
  # ./preprocess.bash -a ./yaml_configs/jijinews/css_srcannt${p}/css_srcannt${p}_unigram.yml
  # ./preprocess.bash ./yaml_configs/jijinews/css_srcannt${p}_inline/css_srcannt${p}_inline_unigram.yml
  # ./preprocess.bash -a ./yaml_configs/jijinews/css_srcannt${p}_inline/css_srcannt${p}_inline_unigram.yml

  # for k in ${ks[@]}; do
  #   ./preprocess.bash -t ./yaml_configs/jijinews/css_srcannt${p}/test_css_srcannt_certain_number${k}.yml
  #   # ./preprocess.bash -ta ./yaml_configs/jijinews/css_srcannt${p}/test_css_srcannt_certain_number${k}.yml
  #   ./preprocess.bash -t ./yaml_configs/jijinews/css_srcannt${p}_inline/test_css_srcannt_certain_number${k}.yml
  #   # ./preprocess.bash -ta ./yaml_configs/jijinews/css_srcannt${p}_inline/test_css_srcannt_certain_number${k}.yml
  # done

  # for base in ${bases[@]}; do
  #   # ./preprocess.bash -t ./yaml_configs/jijinews/css_srcannt${p}_inline/test_baobab_${base}.yml
  #   ./preprocess.bash -t ./yaml_configs/jijinews/css_srcannt${p}/test_baobab_${base}.yml
  # done
done

for k in ${ks[@]}; do
  # ./preprocess.bash -t ./yaml_configs/jijinews/css_srcannt_synthetic/test_css_srcannt_certain_number${k}.yml
  # ./preprocess.bash -t ./yaml_configs/jijinews/css_srcannt_inline_synthetic/test_css_srcannt_certain_number${k}.yml
done

for base in ${bases[@]}; do
  # ./preprocess.bash -t ./yaml_configs/jijinews/css_srcannt_synthetic/test_baobab_${base}.yml
  # ./preprocess.bash -t ./yaml_configs/jijinews/css_srcannt_inline_synthetic/test_baobab_${base}.yml
  ./preprocess.bash -t ./yaml_configs/jijinews/css_srcannt010/test_baobab_${base}.yml
done
